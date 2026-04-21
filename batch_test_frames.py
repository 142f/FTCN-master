import os
import json
import re
import gc
import heapq
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss,
    average_precision_score, f1_score
)

from utils.plugin_loader import PluginLoader
from config import config as cfg


FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LABEL_TOKEN_RE = re.compile(r"(^|[^a-z])(real|fake)([^a-z]|$)")
DEFAULT_OUTPUT_DIR = "test_results_auto"
DEFAULT_INPUT_CANDIDATES = (
    os.environ.get("FTCN_INPUT_DIR", "").strip(),
    r"E:\data\LAV-DF-test-pre",
    r"E:\data\LAV-DF-pre",
    r"E:\data\FakeAVCeleb-test-pre",
    "tmp_nested_input",
)

_RUNTIME_DEPS = None
_NORM_CACHE = {}                                          # [PERF-1]
_FRAME_FOLDER_CACHE = {}


def _is_cuda_oom(exc):
    """Best-effort detection for CUDA OOM runtime errors."""
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        and ("cuda" in msg or "cudnn" in msg or "cublas" in msg)
    )


# -----------------------------------------------------------------
# [PERF-1] 按 device 缓存 ImageNet 归一化张量，避免每次推理重建
# -----------------------------------------------------------------
def _get_norm_tensors(device):
    key = str(device)
    if key not in _NORM_CACHE:
        mean = torch.tensor(
            [0.485 * 255, 0.456 * 255, 0.406 * 255], device=device
        ).view(1, 3, 1, 1, 1)
        std = torch.tensor(
            [0.229 * 255, 0.224 * 255, 0.225 * 255], device=device
        ).view(1, 3, 1, 1, 1)
        _NORM_CACHE[key] = (mean, std)
    return _NORM_CACHE[key]


def get_runtime_deps():
    """Lazy-load heavy detector/alignment dependencies."""
    global _RUNTIME_DEPS
    if _RUNTIME_DEPS is not None:
        return _RUNTIME_DEPS

    print("正在加载检测与对齐依赖（首次可能需要 10-60 秒）...")
    try:
        from test_tools.common import detector, get_lm68
        from test_tools.ct.operations import find_longest, multiple_tracking
        from test_tools.utils import get_crop_box, flatten, partition
        from test_tools.faster_crop_align_xray import FasterCropAlignXRay
        from test_tools.ct.detection.utils import get_valid_faces
    except KeyboardInterrupt as e:
        raise SystemExit(
            "初始化检测依赖时被中断，请稍后重试或避免在首次加载阶段 Ctrl+C。"
        ) from e

    _RUNTIME_DEPS = {
        "detector": detector,
        "get_lm68": get_lm68,
        "find_longest": find_longest,
        "multiple_tracking": multiple_tracking,
        "get_crop_box": get_crop_box,
        "flatten": flatten,
        "partition": partition,
        "FasterCropAlignXRay": FasterCropAlignXRay,
        "get_valid_faces": get_valid_faces,
    }
    return _RUNTIME_DEPS


def to_fake_confidence(score, model_output_semantics="fake"):
    """Convert model score into fake confidence in [0, 1]."""
    score = extract_scalar_score(score)
    if model_output_semantics == "real":
        score = 1.0 - score
    return float(np.clip(score, 0.0, 1.0))


def extract_scalar_score(score):
    """Normalize model output into a Python float scalar."""
    if isinstance(score, torch.Tensor):
        if score.numel() == 0:
            raise ValueError("empty tensor score")
        return float(score.detach().float().reshape(-1)[0].item())

    if isinstance(score, np.ndarray):
        if score.size == 0:
            raise ValueError("empty ndarray score")
        return float(score.reshape(-1)[0])

    if isinstance(score, (list, tuple)):
        if len(score) == 0:
            raise ValueError("empty list/tuple score")
        return float(np.asarray(score).reshape(-1)[0])

    return float(score)


def infer_label_from_path(path):
    """Infer label from path segments using token-aware matching.

    Iterate from leaf to root to prioritise the closest label-like folder name
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


def get_cached_frame_folders(input_dir):
    """Collect frame folders once per absolute input directory."""
    abs_input_dir = os.path.abspath(input_dir)
    cached = _FRAME_FOLDER_CACHE.get(abs_input_dir)
    if cached is None:
        # OPTIM: reuse recursive scan results so probing and execution do not
        # each walk the same directory tree.
        cached = collect_frame_folders(abs_input_dir)
        _FRAME_FOLDER_CACHE[abs_input_dir] = cached
    return cached


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
        if get_cached_frame_folders(abs_candidate):
            return abs_candidate

    return None


def load_frames_from_folder(frame_folder, max_frame=0):
    """极速图像加载器（兼容中文/多字节路径）。"""
    with os.scandir(frame_folder) as entries:
        frame_iter = (
            entry.path
            for entry in entries
            if entry.name.lower().endswith(FRAME_EXTS)
        )
        if max_frame and max_frame > 0:
            # OPTIM: only select the first N sorted frame names when max_frame
            # is active, avoiding a full sort for very large folders.
            frame_files = heapq.nsmallest(max_frame, frame_iter)
        else:
            frame_files = sorted(frame_iter)

    frames = []
    for frame_file in frame_files:
        try:
            # 使用 np.fromfile + imdecode 兼容 Windows 中文路径
            img_data = np.fromfile(frame_file, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("文件数据损坏或格式不支持")
            frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            tqdm.write(f"  [跳过损坏图片] {frame_file}: {e}")
    return frames


def process_single_video_frame_folder(
    frame_folder,
    classifier,
    crop_align_func,
    device,
    max_frame=768,
    model_output_semantics="fake",
    runtime_deps=None,
    batch_size=8,
):
    """处理单视频帧目录，输出视频级和帧级预测得分（Batch 推理）。"""
    mean, std = _get_norm_tensors(device)
    batch_size = max(1, int(batch_size))

    frames = load_frames_from_folder(frame_folder, max_frame=max_frame)
    if len(frames) == 0:
        return None, None

    deps = runtime_deps or get_runtime_deps()

    # ---- Face detection ----
    try:
        with torch.no_grad():
            detect_res = deps["flatten"](
                [deps["detector"].detect(item)
                 for item in deps["partition"](frames, 50)]
            )
            detect_res = deps["get_valid_faces"](detect_res, thres=0.5)
            all_lm68 = deps["get_lm68"](frames, detect_res)
    except RuntimeError as e:
        if _is_cuda_oom(e):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            tqdm.write(
                f"  [检测阶段 OOM] {os.path.basename(frame_folder)}: "
                "显存不足，已跳过该样本。"
            )
            return None, None
        tqdm.write(
            f"  [检测阶段失败] {os.path.basename(frame_folder)}: {e}"
        )
        return None, None
    except Exception as e:
        tqdm.write(
            f"  [检测阶段失败] {os.path.basename(frame_folder)}: {e}"
        )
        return None, None

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
    # OPTIM: 68-point landmarks are already folded into detect_res tuples.
    del all_lm68, all_detect_res

    # ---- Tracking ----
    tracks = deps["multiple_tracking"](detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    if len(tracks) == 0:
        tuples, tracks = deps["find_longest"](detect_res)

    data_storage = {}
    clip_track_sizes = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        if len(track) == 0:
            continue
        clip_track_sizes.append((track_i, len(track)))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = deps["get_crop_box"](shape, box, scale=0.5)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = map(int, big_box)
            if x2 <= x1 or y2 <= y1:
                continue
            cropped = frames[frame_idx][y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            # 避免高频 f-string key 带来的哈希与 GC 负担
            data_storage[(track_i, j, "img")] = cropped
            data_storage[(track_i, j, "ldm")] = info
            data_storage[(track_i, j, "idx")] = frame_idx

    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for track_i, super_clip_size in clip_track_sizes:
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:
            if super_clip_size <= 2:
                continue
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(pre_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)
        frame_range = [
            inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
        ]
        for indices in frame_range:
            clip = [(track_i, t) for t in indices]
            clips_for_video.append(clip)

    if len(clips_for_video) == 0:
        return None, None

    # 原始帧数组体量较大，后续流程不再需要
    del frames

    # ---- Batch Inference ----
    preds = []
    frame_res = {}
    batch_clips = []
    batch_frame_ids = []
    current_batch_limit = batch_size

    def infer_with_auto_split(clips, frame_ids):
        nonlocal current_batch_limit
        if not clips:
            return

        try:
            with torch.no_grad():
                batch_input = torch.stack(clips, dim=0).to(device)
                batch_input = batch_input.sub(mean).div(std)
                batch_output = classifier(batch_input)["final_output"]
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            clip_count = len(clips)
            if clip_count <= 1:
                tqdm.write(
                    f"  [推理阶段 OOM] {os.path.basename(frame_folder)}: "
                    "单 clip 仍显存不足，已跳过该 clip。"
                )
                return

            new_limit = max(1, clip_count // 2)
            if new_limit < current_batch_limit:
                current_batch_limit = new_limit
                tqdm.write(
                    f"  [推理阶段 OOM] {os.path.basename(frame_folder)}: "
                    f"自动降批到 {current_batch_limit}。"
                )

            mid = clip_count // 2
            infer_with_auto_split(clips[:mid], frame_ids[:mid])
            infer_with_auto_split(clips[mid:], frame_ids[mid:])
            return

        for local_idx, frame_ids_one_clip in enumerate(frame_ids):
            pred = to_fake_confidence(
                batch_output[local_idx], model_output_semantics
            )
            preds.append(pred)
            for f_id in frame_ids_one_clip:
                frame_res.setdefault(f_id, []).append(pred)

        del batch_input, batch_output

    def flush_batch():
        nonlocal batch_clips, batch_frame_ids
        if not batch_clips:
            return

        infer_with_auto_split(batch_clips, batch_frame_ids)

        batch_clips = []
        batch_frame_ids = []

    for clip in clips_for_video:
        try:
            images = [data_storage[(i, j, "img")] for i, j in clip]
            landmarks = [data_storage[(i, j, "ldm")] for i, j in clip]
            frame_ids = [data_storage[(i, j, "idx")] for i, j in clip]
        except KeyError:
            continue

        try:
            landmarks, images = crop_align_func(landmarks, images)
        except Exception as e:
            if frame_folder:
                tqdm.write(f"  [crop_align 失败] {os.path.basename(frame_folder)}: {e}")
            continue

        images_tensor = torch.as_tensor(images, dtype=torch.float32)
        images_tensor = images_tensor.permute(3, 0, 1, 2).contiguous()
        batch_clips.append(images_tensor)
        batch_frame_ids.append(frame_ids)

        if len(batch_clips) >= current_batch_limit:
            flush_batch()

    flush_batch()
    del data_storage, clips_for_video

    if len(preds) == 0:
        return None, None

    video_level_score = float(np.mean(preds))

    # 使用 sorted(frame_res) 替代 range(len(frames))（frames 已释放）
    frame_level_scores = [
        {
            "frame_id": fid,
            "fake_confidence": float(np.mean(frame_res[fid])),
        }
        for fid in sorted(frame_res)
    ]

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
        default="fake",
        help=(
            "模型 output['final_output'] 的语义。"
            "real=真脸置信度(转换为fake_confidence=1-score)，"
            "fake=伪造置信度(直接使用)。"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="每次前向推理的 clip 批大小（按显存调整，OOM 时会自动降批）",
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
    print(f"model_output_semantics(effective): {args.model_output_semantics}")

    # ---- Load model ----
    cfg.init_with_yaml()
    cfg.update_with_yaml("ftcn_tt.yaml")
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier = classifier.to(device)
    classifier.eval()
    classifier.load("checkpoints/ftcn_tt.pth")

    runtime_deps = get_runtime_deps()
    crop_align_func = runtime_deps["FasterCropAlignXRay"](cfg.imsize)

    os.makedirs(args.out_dir, exist_ok=True)

    video_list_all = get_cached_frame_folders(args.input_dir)

    video_list = video_list_all
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
                runtime_deps=runtime_deps,
                batch_size=args.batch_size,
            )

            if vid_score is None:
                tqdm.write(
                    f"  Skipping {v_name}: "
                    f"No faces detected or clips too short."
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
    preds_csv_path = os.path.join(
        args.out_dir, "batch_predictions_summary.csv"
    )
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"\n所有视频的预测得分已汇总保存至: {preds_csv_path}")

    # ---- Metrics ----
    if len(df_preds) == 0:
        print("没有可用于统计的预测结果。")
    else:
        y_true = df_preds["true_label"].values
        y_pred = df_preds["pred_score"].values
        has_both_classes = len(set(y_true)) > 1

        try:
            if has_both_classes:
                auc = roc_auc_score(y_true, y_pred)
                ap = average_precision_score(y_true, y_pred)
                logloss = log_loss(y_true, y_pred, labels=[0, 1])
            else:
                auc = ap = logloss = float("nan")

            y_pred_binary = (y_pred >= args.eval_threshold).astype(int)
            acc = accuracy_score(y_true, y_pred_binary)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)

            print("=" * 40)
            print("【量化测试指标结果】")
            print(f"参与计算样本总数 : {len(y_true)}")
            print(
                f"AUC Score     : {auc:.4f}"
                if not np.isnan(auc)
                else "AUC Score     : N/A (单类别)"
            )
            print(
                f"AP Score      : {ap:.4f}"
                if not np.isnan(ap)
                else "AP Score      : N/A (单类别)"
            )
            print(
                f"Accuracy (ACC) : {acc:.4f} "
                f"(阈值{args.eval_threshold})"
            )
            print(
                f"F1 Score      : {f1:.4f} "
                f"(阈值{args.eval_threshold})"
            )
            print(
                f"Log Loss      : {logloss:.4f}"
                if not np.isnan(logloss)
                else "Log Loss      : N/A (单类别)"
            )
            print("=" * 40)

        except Exception as e:
            print(f"[严重异常] 计算评估指标时出错: {e}")
