import os
import glob
import json
import re
import gc                                                            # [FIX-7]
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss,
    average_precision_score, f1_score
)

from utils.plugin_loader import PluginLoader
from config import config as cfg
from test_tools.common import detector, get_lm68
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.utils import get_crop_box, flatten, partition
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.ct.detection.utils import get_valid_faces


FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_TOKEN_RE = re.compile(r"(^|[^a-z])(real|fake)([^a-z]|$)")
DEFAULT_OUTPUT_DIR = "test_results_auto"
DEFAULT_INPUT_CANDIDATES = (
    os.environ.get("FTCN_INPUT_DIR", "").strip(),
    r"E:\data\LAV-DF-pre",
    r"E:\data\FakeAVCeleb-test-pre",
    "tmp_nested_input",
)


def to_fake_confidence(score, model_output_semantics="real"):
    """Convert model score into fake confidence in [0, 1]."""
    score = float(score)
    if model_output_semantics == "real":
        score = 1.0 - score
    return float(np.clip(score, 0.0, 1.0))


def infer_label_from_path(path):
    """Infer label from path segments using token-aware matching.

    Iterate from leaf to root to prioritize the closest label-like folder name
    (e.g. '0_real', '1_fake') over dataset names that incidentally contain
    'fake' in the middle (e.g. 'FakeAVCeleb').
    """
    for part in reversed(os.path.normpath(path).split(os.sep)):
        lower_part = part.lower()
        token_match = LABEL_TOKEN_RE.search(lower_part)
        if not token_match:
            continue
        token = token_match.group(2)
        if token == "fake":
            return 1
        if token == "real":
            return 0
    return None


def collect_frame_folders(input_dir):
    """Recursively collect folders that directly contain image frames."""
    video_list = []
    seen = set()

    for root, _, files in os.walk(input_dir):
        label = infer_label_from_path(root)
        if label is None:
            continue

        has_frame = any(
            file_name.lower().endswith(FRAME_EXTS) for file_name in files
        )
        if not has_frame:
            continue

        norm_root = os.path.normpath(root)
        if norm_root in seen:
            continue

        seen.add(norm_root)
        video_list.append({"path": root, "label": label})

    video_list.sort(key=lambda x: x["path"])
    return video_list


def resolve_input_dir(input_dir):
    """Resolve CLI input_dir or pick the first usable default dataset path."""
    if input_dir:
        candidate = os.path.abspath(input_dir)
        if os.path.isdir(candidate):
            return candidate
        return None

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if not candidate:
            continue
        abs_candidate = os.path.abspath(candidate)
        if not os.path.isdir(abs_candidate):
            continue
        if len(collect_frame_folders(abs_candidate)) > 0:
            return abs_candidate

    return None


def load_frames_from_folder(frame_folder):
    """Load all image frames from a folder, sorted by filename."""
    exts = tuple(f"*{ext}" for ext in FRAME_EXTS)
    frame_files = []
    for ext in exts:
        frame_files.extend(glob.glob(os.path.join(frame_folder, ext)))
    frame_files = sorted(frame_files)

    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        frames.append(np.array(img))
    return frames


def process_single_video_frame_folder(
    frame_folder,
    classifier,
    crop_align_func,
    device,
    max_frame=768,
    model_output_semantics="real",
):
    """Process a single video (frame folder) through the FTCN pipeline.

    Returns:
        (video_level_score, frame_level_scores) or (None, None) on failure.
    """
    mean = torch.tensor(
        [0.485 * 255, 0.456 * 255, 0.406 * 255], device=device
    ).view(1, 3, 1, 1, 1)
    std = torch.tensor(
        [0.229 * 255, 0.224 * 255, 0.225 * 255], device=device
    ).view(1, 3, 1, 1, 1)

    frames = load_frames_from_folder(frame_folder)
    if len(frames) == 0:
        return None, None

    if len(frames) > max_frame:
        frames = frames[:max_frame]

    # ---- Face detection ----
    detect_res = flatten(
        [detector.detect(item) for item in partition(frames, 50)]
    )
    detect_res = get_valid_faces(detect_res, thres=0.5)
    all_lm68 = get_lm68(frames, detect_res)

    if len(detect_res) == 0:
        return None, None

    shape = frames[0].shape[:2]
    all_detect_res = []

    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)

    detect_res = all_detect_res

    # ---- Tracking ----
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    super_clips = []
    # [FIX-2] 记录每个有效轨迹对应的真实 track_i，避免索引错位
    valid_track_indices = []

    for track_i, ((start, end), track) in enumerate(
        zip(tuples, tracks)
    ):
        if len(track) == 0:
            continue

        # [FIX-2] 保存真实 track_i，后续用它做 data_storage 的键前缀
        valid_track_indices.append(track_i)
        super_clips.append(len(track))

        for face, frame_idx, j in zip(
            track, range(start, end), range(len(track))
        ):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]

            # [FIX-2] 键前缀始终使用 track_i（与后续查找一致）
            base_key = f"{track_i}_{j}_"
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx
            # [FIX-4] 移除从未使用的 frame_boxes 计算

    # ---- Build clips ----
    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    # [FIX-2] 使用 valid_track_indices 让键前缀与 data_storage 一致
    for valid_ti, super_clip_size in zip(valid_track_indices, super_clips):
        inner_index = list(range(super_clip_size))

        if super_clip_size < clip_size:
            # 短于 clip_size 的轨迹需要做镜像填充
            if super_clip_size <= 2:
                continue

            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]

            pre_module = inner_index + inner_index[1:-1][::-1]
            # [FIX-1] 修复复制粘贴错误：应测量 pre_module 自身长度
            l_pre = len(pre_module)    # ← 原代码误写为 len(post_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]

            inner_index = pre_module + inner_index + post_module

        total_len = len(inner_index)
        frame_range = [
            inner_index[i : i + clip_size]
            for i in range(total_len)
            if i + clip_size <= total_len
        ]

        for indices in frame_range:
            # [FIX-2] 用 valid_ti 而非连续递增的 super_clip_idx
            clips_for_video.append([(valid_ti, t) for t in indices])

    if len(clips_for_video) == 0:
        return None, None

    # ---- Inference ----
    preds = []
    frame_res = {}

    for clip in clips_for_video:
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        # [FIX-3] 记录对齐失败而非静默吞掉
        try:
            landmarks, images = crop_align_func(landmarks, images)
        except Exception as e:
            if frame_folder:  # 避免空路径
                tqdm.write(
                    f"  [crop_align 失败] {os.path.basename(frame_folder)}: {e}"
                )
            continue

        images = torch.as_tensor(
            images, dtype=torch.float32, device=device
        ).permute(3, 0, 1, 2)
        images = images.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images)

        pred = to_fake_confidence(
            output["final_output"], model_output_semantics
        )
        preds.append(pred)

        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)

    if len(preds) == 0:
        return None, None

    video_level_score = float(np.mean(preds))

    frame_level_scores = []
    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            frame_level_scores.append(
                {
                    "frame_id": frame_idx,
                    "fake_confidence": float(np.mean(frame_res[frame_idx])),
                }
            )

    return video_level_score, frame_level_scores


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="帧目录，支持递归查找 real/fake 下的帧子目录",
    )
    parser.add_argument(
        "--out_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录"
    )
    parser.add_argument(
        "--max_frame", type=int, default=768, help="每个样本最多处理帧数"
    )
    parser.add_argument(
        "--max_videos", type=int, default=0, help="最多处理样本数，0=全部"
    )
    parser.add_argument(
        "--eval_threshold", type=float, default=0.5, help="ACC/F1 阈值"
    )
    parser.add_argument(
        "--model_output_semantics",
        type=str,
        choices=["real", "fake"],
        default="real",
        help=(
            "模型 output['final_output'] 的语义。"
            "real=真脸置信度(转换为fake_confidence=1-score)，"
            "fake=伪造置信度(直接使用)。"
        ),
    )
    args = parser.parse_args()

    # ---- Resolve input ----
    user_input_dir = args.input_dir
    resolved_input_dir = resolve_input_dir(user_input_dir)
    if resolved_input_dir is None:
        raise FileNotFoundError(
            "未找到可用输入目录。请通过 --input_dir 显式指定。"
        )

    args.input_dir = resolved_input_dir
    if not user_input_dir:
        print(f"未传 --input_dir，自动使用: {args.input_dir}")

    if not args.out_dir:
        args.out_dir = DEFAULT_OUTPUT_DIR
        print(f"未传 --out_dir，自动使用: {args.out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    print(f"model_output_semantics: {args.model_output_semantics}")

    # ---- Load model ----
    cfg.init_with_yaml()
    cfg.update_with_yaml("ftcn_tt.yaml")
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier = classifier.to(device)
    classifier.eval()
    classifier.load("checkpoints/ftcn_tt.pth")

    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    os.makedirs(args.out_dir, exist_ok=True)

    video_list = collect_frame_folders(args.input_dir)
    if args.max_videos > 0:
        video_list = video_list[: args.max_videos]

    print(f"找到 {len(video_list)} 个帧文件夹准备处理...")

    if len(video_list) == 0:
        print(
            "未找到可处理帧目录。请确认 input_dir 下存在 real/fake 路径，"
            "并且目标目录内包含图片帧文件。"
        )

    video_results = []

    for v_info in tqdm(video_list, desc="Batch Processing"):
        v_path = v_info["path"]
        v_name = os.path.basename(v_path)
        true_label = v_info["label"]

        try:
            vid_score, frame_scores = process_single_video_frame_folder(
                v_path,
                classifier,
                crop_align_func,
                device,
                max_frame=args.max_frame,
                model_output_semantics=args.model_output_semantics,
            )

            if vid_score is None:
                tqdm.write(
                    f"  Skipping {v_name}: No faces detected or clips too short."
                )
                continue

            video_results.append(
                {
                    "video_name": v_name,
                    "pred_score": vid_score,
                    "true_label": true_label,
                }
            )

            sub_dir_name = "fake" if true_label == 1 else "real"
            video_out_dir = os.path.join(args.out_dir, sub_dir_name)
            os.makedirs(video_out_dir, exist_ok=True)

            json_path = os.path.join(
                video_out_dir, f"{v_name}_frame_preds.json"
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(frame_scores, f, indent=2)

        except Exception as e:
            tqdm.write(f"  Error processing {v_name}: {e}")

        # [FIX-7] 每个视频处理完后释放 GPU 显存碎片
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ---- Save predictions ----
    df_preds = pd.DataFrame(video_results)
    preds_csv_path = os.path.join(args.out_dir, "batch_predictions_summary.csv")
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"\n所有视频的预测得分已汇总保存至: {preds_csv_path}")

    # ---- Metrics ----
    if len(df_preds) == 0:
        print("没有可用于统计的预测结果。")
    else:
        y_true = df_preds["true_label"].values
        y_pred = df_preds["pred_score"].values
        has_both_classes = len(set(y_true)) > 1       # [FIX-6]

        try:
            if has_both_classes:
                auc = roc_auc_score(y_true, y_pred)
                ap = average_precision_score(y_true, y_pred)
            else:
                auc = float("nan")
                ap = float("nan")

            y_pred_binary = (y_pred >= args.eval_threshold).astype(int)
            acc = accuracy_score(y_true, y_pred_binary)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)

            # [FIX-6] log_loss 在单类别时跳过（数学上无意义）
            if has_both_classes:
                logloss = log_loss(y_true, y_pred, labels=[0, 1])
            else:
                logloss = float("nan")

            print("=" * 40)
            print("【量化测试指标结果】")
            print(f"参与计算样本总数 : {len(y_true)}")
            print(
                f"AUC Score     : {auc:.4f}"
                if not np.isnan(auc)
                else "AUC Score     : N/A (单类别)"
            )
            print(
                f"Average Precision (AP) : {ap:.4f}"
                if not np.isnan(ap)
                else "AP Score      : N/A (单类别)"
            )
            print(f"Accuracy (ACC) : {acc:.4f} (阈值{args.eval_threshold})")
            print(f"F1 Score      : {f1:.4f} (阈值{args.eval_threshold})")
            print(
                f"Log Loss      : {logloss:.4f}"
                if not np.isnan(logloss)
                else "Log Loss      : N/A (单类别)"
            )

            if has_both_classes:
                thresholds = np.linspace(0.0, 1.0, 201)
                f1_list = [
                    f1_score(
                        y_true, (y_pred >= th).astype(int), zero_division=0
                    )
                    for th in thresholds
                ]
                best_idx = int(np.argmax(f1_list))
                best_th = float(thresholds[best_idx])
                best_f1 = float(f1_list[best_idx])
                print(f"Best F1      : {best_f1:.4f} (阈值{best_th:.3f})")

            if not np.isnan(auc) and auc < 0.5:
                print(
                    "提示：AUC < 0.5，分数方向可能与标签定义相反。"
                    "可尝试切换 --model_output_semantics 为另一种取值。"
                )
            print("=" * 40)

        except Exception as e:
            print(f"计算评估指标时出错: {e}")