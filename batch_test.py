import os
import glob
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, average_precision_score, f1_score

from utils.plugin_loader import PluginLoader
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay

def process_single_video(input_file, classifier, crop_align_func, max_frame=768):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

    cache_file = f"{input_file}_{str(max_frame)}.pth"

    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(input_file, max_size=max_frame, cvt=True)
    else:
        detect_res, all_lm68, frames = detect_all(
            input_file, return_frames=True, max_size=max_frame
        )
        torch.save((detect_res, all_lm68), cache_file)

    if len(frames) == 0 or len(detect_res) == 0:
        return None, None

    shape = frames[0].shape[:2]
    all_detect_res = []
    
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_face = (box, lm5, face_lm68, score)
            new_faces.append(new_face)
        all_detect_res.append(new_faces)

    detect_res = all_detect_res
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        if len(track) == 0: continue
        super_clips.append(len(track))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]

            base_key = f"{track_i}_{j}_"
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(np.int64)

    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:
            if super_clip_size <= 2: continue # skip drastically small clips
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(post_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)
        frame_range = [
            inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)
    
    if len(clips_for_video) == 0:
        return None, None

    preds = []
    frame_res = {}

    for clip in clips_for_video:
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
        
        try:
            landmarks, images = crop_align_func(landmarks, images)
        except Exception:
            continue
            
        images = torch.as_tensor(images, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
        images = images.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images)

        pred = float(output["final_output"])
        preds.append(pred)
        
        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)

    if len(preds) == 0:
         return None, None
         
    video_level_score = np.mean(preds)
    
    # 获取纯帧级别预测结果 (将同一个frame所在所有clip的预测score取平均)
    frame_level_scores = []
    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            frame_pred = float(np.mean(frame_res[frame_idx]))
            frame_level_scores.append({
                "frame_id": frame_idx,
                "fake_confidence": frame_pred
            })
            
    return video_level_score, frame_level_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="大文件夹目录，包含 'real' 和 'fake' 两个子文件夹，或在此文件夹内直接放视频。")
    parser.add_argument("--out_dir", type=str, required=True, 
                        help="保存预测结果的输出目录。")
    # 可选：如果有一个专门的CSV记录每个视频的真实标签（用于计算AUC等），可以传入
    parser.add_argument("--label_csv", type=str, default=None, 
                        help="包含 [video_name, label] 的真实标签 CSV。如果目录有real/fake子文件夹，则自动生成标签。")

    args = parser.parse_args()

    # 初始化配置和模型
    cfg.init_with_yaml()
    cfg.update_with_yaml("ftcn_tt.yaml")
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda()
    classifier.eval()
    classifier.load("checkpoints/ftcn_tt.pth")
    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 搜集所有视频并自动打标签（如果有 real/fake 子文件夹）
    video_list = []  # 格式: {"path": str, "label": int or None}
    
    # 检查是否存在包含 real 或 fake 的子文件夹
    has_subfolders = False
    for subfolder in os.listdir(args.input_dir):
        sub_path = os.path.join(args.input_dir, subfolder)
        if os.path.isdir(sub_path):
            label = None
            if 'real' in subfolder.lower():
                label = 0
            elif 'fake' in subfolder.lower():
                label = 1
                
            if label is not None:
                has_subfolders = True
                exts = ['*.mp4', '*.avi', '*.mov']
                for ext in exts:
                    paths = glob.glob(os.path.join(sub_path, ext))
                    for p in paths:
                        video_list.append({"path": p, "label": label})
    
    # 如果没有 real/fake 子文件夹，直接读取平铺的视频
    if not has_subfolders:
        print("未检测到 'real' 或 'fake' 子文件夹，正在从根目录直接读取所有视频...")
        exts = ['*.mp4', '*.avi', '*.mov']
        for ext in exts:
            paths = glob.glob(os.path.join(args.input_dir, ext))
            for p in paths:
                video_list.append({"path": p, "label": None})

    print(f"找到 {len(video_list)} 个视频准备处理...")

    video_results = []

    # 2. 批量处理视频
    for v_info in tqdm(video_list, desc="Batch Processing"):
        v_path = v_info["path"]
        v_name = os.path.basename(v_path)
        true_label = v_info["label"]
        
        try:
            vid_score, frame_scores = process_single_video(v_path, classifier, crop_align_func)
            
            if vid_score is not None:
                record = {
                    "video_name": v_name,
                    "pred_score": vid_score
                }
                # 如果从子文件夹自动推断出了标签，则加入结果中
                if true_label is not None:
                    record["true_label"] = true_label
                video_results.append(record)
                
                # 为该视频创建独立的存储文件夹，保持结果整洁
                # 真实/伪造文件夹的结果分开放置
                sub_dir_name = "fake" if true_label == 1 else ("real" if true_label == 0 else "unknown")
                video_out_dir = os.path.join(args.out_dir, sub_dir_name)
                os.makedirs(video_out_dir, exist_ok=True)
                
                # 保存该视频的纯帧级预测结果
                json_path = os.path.join(video_out_dir, f"{v_name}_frame_preds.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(frame_scores, f, indent=4)
            else:
                print(f"\nSkipping {v_name}: No faces detected or clips too short.")
        except Exception as e:
            print(f"\nError processing {v_name}: {e}")
            
    # 3. 保存所有视频的总体预测分数（这个CSV包含所有信息，作为统一的结果）
    df_preds = pd.DataFrame(video_results)
    preds_csv_path = os.path.join(args.out_dir, "batch_predictions_summary.csv")
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"\n所有视频的预测得分已汇总保存至: {preds_csv_path}")

    # 4. (可选) 计算可量化的参数 (AUC, ACC, LogLoss)
    y_true = []
    y_pred = []
    
    # 优先使用由于子文件夹自然带上的标签
    if "true_label" in df_preds.columns:
        y_true = df_preds["true_label"].values
        y_pred = df_preds["pred_score"].values
        print("从文件夹结构检测到自动标签。")
    # 其次使用显式提供的CSV
    elif args.label_csv and os.path.exists(args.label_csv):
        df_labels = pd.read_csv(args.label_csv)
        df_merged = pd.merge(df_preds, df_labels, on="video_name", how="inner")
        if not df_merged.empty:
            y_true = df_merged["label"].values # 假设 1 代表 Fake, 0 代表 Real
            y_pred = df_merged["pred_score"].values
            print("从外部CSV加载真实标签。")
        else:
            print("标CSV中没有匹配上处理过的视频名。")
            
    if len(y_true) > 0 and len(y_true) == len(y_pred):
        try:
            # 计算 AUC, 避免全部是一类数据导致报错
            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, y_pred)
                ap = average_precision_score(y_true, y_pred)
            else:
                auc = float('nan')
                ap = float('nan')
                print("\n提示：当前检测到的数据只有一类标签（纯真/纯假），无法计算AUC和AP，需要同时有真假数据。")
            
            # 计算 ACC 和 F1 (以0.5作为阈值)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred_binary)
            f1 = f1_score(y_true, y_pred_binary)
            
            # 计算 Log Loss
            logloss = log_loss(y_true, y_pred)
            
            print("="*40)
            print("【量化测试指标结果】")
            print(f"参与计算样本总数 : {len(y_true)}")
            print(f"AUC Score     : {auc:.4f}" if not np.isnan(auc) else "AUC Score     : N/A")
            print(f"Average Precision (AP) : {ap:.4f}" if not np.isnan(ap) else "AP Score      : N/A")
            print(f"Accuracy (ACC) : {acc:.4f} (阈值0.5)")
            print(f"F1 Score      : {f1:.4f} (阈值0.5)")
            print(f"Log Loss      : {logloss:.4f}")
            print("="*40)
        except Exception as e:
            print(f"计算评价指标时出错: {e}")
    else:
        print("\n未检测到充分的真实和伪造数据标签，无法计算AUC/ACC准确度。")
