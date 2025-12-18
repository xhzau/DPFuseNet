# -*- coding: utf-8 -*-
"""
Evaluate instance segmentation metrics by pairing files from two folders:
- pred_dir: predicted labels (txt with columns: x y z pred_label [>=4 cols, last col used])
- gt_dir  : ground truth labels (txt with columns: x y z gt_label [>=4 cols, last col used])

Files are paired by filename stem (case-sensitive). Rows are assumed aligned (same point order).
Outputs:
  - per_file_metrics.csv (per file)
  - dataset_summary.txt  (aggregated over dataset)

Python 3.8 compatible.
"""

import os
import csv
import glob
import argparse
import numpy as np
from typing import Dict, Optional, Tuple, List

# ----------------------------
# IoU 矩阵与实例列表
# ----------------------------
def _iou_matrix(gt_labels: np.ndarray,
                pred_labels: np.ndarray,
                ignore_label: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 IoU 矩阵及 GT/Pred 实例 ID 和各自点数。"""
    gt = gt_labels.copy()
    pr = pred_labels.copy()

    if ignore_label is not None:
        mask = (gt != ignore_label)
        gt = gt[mask]
        pr = pr[mask]

    gt_ids = np.unique(gt)
    pr_ids = np.unique(pr)
    M, N = len(gt_ids), len(pr_ids)

    iou = np.zeros((M, N), dtype=float)
    gt_sz = np.zeros(M, dtype=int)
    pr_sz = np.zeros(N, dtype=int)

    gt_masks = []
    for i, gid in enumerate(gt_ids):
        m = (gt == gid)
        gt_masks.append(m)
        gt_sz[i] = int(m.sum())

    pr_masks = []
    for j, pid in enumerate(pr_ids):
        m = (pr == pid)
        pr_masks.append(m)
        pr_sz[j] = int(m.sum())

    for i in range(M):
        gm = gt_masks[i]
        # 小优化：只计算交集>0 的列
        for j in range(N):
            pm = pr_masks[j]
            inter = int(np.sum(gm & pm))
            if inter == 0:
                continue
            union = int(np.sum(gm | pm))
            iou[i, j] = inter / union if union > 0 else 0.0

    return iou, gt_ids, pr_ids, gt_sz, pr_sz

# ----------------------------
# mCov / mWCov（无需匹配）
# ----------------------------
def mean_coverage(gt_labels: np.ndarray,
                  pred_labels: np.ndarray,
                  ignore_label: Optional[int] = None
                  ) -> Tuple[float, float]:
    """
    mCov  = (1/|I|) * sum_m  max_n IoU(I_m, P_n)
    mWCov = sum_m (|I_m|/sum_k|I_k|) * max_n IoU(I_m, P_n)
    """
    iou, gt_ids, pr_ids, gt_sz, _ = _iou_matrix(gt_labels, pred_labels, ignore_label)
    M = len(gt_ids)
    if M == 0:
        return 0.0, 0.0
    best_iou_per_gt = iou.max(axis=1) if iou.size > 0 else np.zeros(M, dtype=float)
    mCov = float(np.mean(best_iou_per_gt))
    weights = gt_sz.astype(float) / float(np.sum(gt_sz)) if np.sum(gt_sz) > 0 else np.zeros_like(gt_sz, dtype=float)
    mWCov = float(np.sum(weights * best_iou_per_gt))
    return mCov, mWCov

# ----------------------------
# mPrec / mRec（阈值匹配：贪心）
# ----------------------------
def mean_precision_recall(gt_labels: np.ndarray,
                          pred_labels: np.ndarray,
                          iou_thresh: float = 0.5,
                          ignore_label: Optional[int] = None
                          ) -> Tuple[float, float, Dict[str, int]]:
    """
    单类实例级精确率/召回率（贪心最大 IoU 一一匹配，不用匈牙利）：
      mPrec = TP / |P_ins|
      mRec  = TP / |G_ins|
    """
    iou, gt_ids, pr_ids, _, _ = _iou_matrix(gt_labels, pred_labels, ignore_label)
    M, N = len(gt_ids), len(pr_ids)

    if M == 0 and N == 0:
        return 0.0, 0.0, dict(TP=0, G=0, P=0)
    if M == 0:
        return 0.0, 0.0, dict(TP=0, G=0, P=N)
    if N == 0:
        return 0.0, 0.0, dict(TP=0, G=M, P=0)

    cand_i, cand_j = np.where(iou >= iou_thresh)
    if len(cand_i) == 0:
        return 0.0, 0.0, dict(TP=0, G=M, P=N)

    cand_vals = iou[cand_i, cand_j]
    order = np.argsort(-cand_vals)  # 按 IoU 降序

    used_i = set()
    used_j = set()
    TP = 0
    for idx in order:
        i = int(cand_i[idx]); j = int(cand_j[idx])
        if i in used_i or j in used_j:
            continue
        TP += 1
        used_i.add(i); used_j.add(j)

    G = M
    P = N
    mPrec = TP / P if P > 0 else 0.0
    mRec  = TP / G if G > 0 else 0.0
    return mPrec, mRec, dict(TP=TP, G=G, P=P)

# ----------------------------
# 汇总接口（单文件）
# ----------------------------
def evaluate_instance_metrics(gt_labels: np.ndarray,
                              pred_labels: np.ndarray,
                              iou_thresh: float = 0.5,
                              ignore_label: Optional[int] = None
                              ) -> Dict[str, float]:
    mCov, mWCov = mean_coverage(gt_labels, pred_labels, ignore_label)
    mPrec, mRec, cnt = mean_precision_recall(gt_labels, pred_labels, iou_thresh, ignore_label)
    return dict(
        mCov=mCov, mWCov=mWCov,
        mPrec=mPrec, mRec=mRec,
        TP=cnt['TP'], G=cnt['G'], P=cnt['P'],
        iou_thresh=iou_thresh
    )

# ----------------------------
# 数据集聚合所需的覆盖分量
# ----------------------------
def _coverage_components_for_agg(gt_labels: np.ndarray,
                                 pred_labels: np.ndarray,
                                 ignore_label: Optional[int] = None
                                 ) -> Tuple[float, int, float, int]:
    """
    返回：
      sum_best_iou : 所有 GT 实例的 best IoU 之和
      total_gt_inst: GT 实例总数
      sum_weighted : sum_m (|I_m| * best_iou_m)
      total_gt_pts : 所有 GT 点总数
    """
    iou, gt_ids, pr_ids, gt_sz, _ = _iou_matrix(gt_labels, pred_labels, ignore_label)
    M = len(gt_ids)
    if M == 0:
        return 0.0, 0, 0.0, 0
    best_iou = iou.max(axis=1) if iou.size > 0 else np.zeros(M, dtype=float)
    sum_best = float(np.sum(best_iou))
    sum_weighted = float(np.sum(best_iou * gt_sz.astype(float)))
    total_pts = int(np.sum(gt_sz))
    return sum_best, M, sum_weighted, total_pts

# ----------------------------
# 读取文件 & 对齐
# ----------------------------
def _read_last_label_col(fp: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 .txt 并返回 (coords, labels)，其中 labels 取最后一列。
    要求 >=4 列；默认前三列为坐标，用于基本一致性检查。
    """
    arr = np.loadtxt(fp)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("文件至少需要四列（x y z label）：{}".format(fp))
    coords = arr[:, :3].astype(float)
    labels = arr[:, -1].astype(int)  # 最后一列作为标签
    return coords, labels

def _read_last_label_col_2(fp: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 .txt 并返回 (coords, labels)，其中 labels 取最后一列。
    要求 >=4 列；默认前三列为坐标，用于基本一致性检查。
    """
    arr = np.loadtxt(fp)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("文件至少需要四列（x y z label）：{}".format(fp))
    coords = arr[:, :3].astype(float)
    labels = arr[:, -1].astype(int)  # 最后一列作为标签
    return coords, labels


def _pair_files(pred_dir: str, gt_dir: str, pattern: str = "*.txt") -> List[Tuple[str,str]]:
    """
    按同名（stem）配对 pred_dir 与 gt_dir 下的文件。
    返回列表 [(pred_path, gt_path), ...]，仅包含两侧都存在的文件。
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, pattern)))
    gt_files   = sorted(glob.glob(os.path.join(gt_dir, pattern)))
    pred_map = {os.path.splitext(os.path.basename(p))[0]: p for p in pred_files}
    gt_map   = {os.path.splitext(os.path.basename(g))[0]: g for g in gt_files}
    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    pairs = [(pred_map[k], gt_map[k]) for k in common]
    return pairs

# ----------------------------
# 主流程：两个目录评估
# ----------------------------
def evaluate_two_folders(pred_dir: str,
                         gt_dir: str,
                         output_dir: str,
                         iou_thresh: float = 0.5,
                         ignore_label: Optional[int] = None,
                         pattern: str = "*.txt",
                         check_xyz: bool = True,
                         xyz_tol: float = 1e-6) -> None:
    """
    将 pred_dir 与 gt_dir 下的同名 txt 文件配对，计算逐文件与数据集指标。
    - 若 check_xyz=True，会检查坐标是否逐行一致（允许 xyz_tol 误差）。不一致则发出警告。
    """
    os.makedirs(output_dir, exist_ok=True)
    pairs = _pair_files(pred_dir, gt_dir, pattern)
    if not pairs:
        raise ValueError("未找到可配对的文件。请确认两目录下存在同名 .txt 文件。")

    csv_path = os.path.join(output_dir, "per_file_metrics.csv")
    txt_path = os.path.join(output_dir, "dataset_summary.txt")

    # 数据集累加器
    sum_best_iou_all = 0.0
    total_gt_inst_all = 0
    sum_weighted_all = 0.0
    total_gt_pts_all = 0

    TP_all = 0
    G_all  = 0
    P_all  = 0

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["file", "mCov", "mWCov", "mPrec", "mRec", "TP", "G", "P", "IoU_thr"])

        for pred_fp, gt_fp in pairs:
            name = os.path.basename(pred_fp)
            try:
                pred_xyz, pred_lab = _read_last_label_col(pred_fp)
                gt_xyz,   gt_lab   = _read_last_label_col_2(gt_fp)

                if pred_xyz.shape[0] != gt_xyz.shape[0]:
                    print(f"[WARN] 点数不一致：{name}  pred={pred_xyz.shape[0]} vs gt={gt_xyz.shape[0]}（跳过坐标一致性检查）")
                elif check_xyz:
                    same = np.allclose(pred_xyz, gt_xyz, atol=xyz_tol)
                    if not same:
                        print(f"[WARN] 坐标不一致（逐行）：{name}（将继续按行评估标签）")

                # 文件级指标
                m = evaluate_instance_metrics(gt_lab, pred_lab, iou_thresh=iou_thresh, ignore_label=ignore_label)
                writer.writerow([name, f"{m['mCov']:.6f}", f"{m['mWCov']:.6f}",
                                 f"{m['mPrec']:.6f}", f"{m['mRec']:.6f}",
                                 int(m['TP']), int(m['G']), int(m['P']), iou_thresh])

                # 覆盖度聚合分量
                sb, Mi, sw, pts = _coverage_components_for_agg(gt_lab, pred_lab, ignore_label=ignore_label)
                sum_best_iou_all += sb
                total_gt_inst_all += Mi
                sum_weighted_all += sw
                total_gt_pts_all += pts

                # P/R 聚合计数
                TP_all += int(m['TP'])
                G_all  += int(m['G'])
                P_all  += int(m['P'])

                print(f"[OK] {name}  mCov={m['mCov']:.4f}  mWCov={m['mWCov']:.4f}  "
                      f"Prec={m['mPrec']:.4f}  Rec={m['mRec']:.4f}  "
                      f"TP={m['TP']} G={m['G']} P={m['P']}")
            except Exception as e:
                print(f"[ERROR] {name}: {e}")

    # 数据集汇总
    if total_gt_inst_all > 0:
        mCov_all  = sum_best_iou_all / float(total_gt_inst_all)
    else:
        mCov_all = 0.0
    if total_gt_pts_all > 0:
        mWCov_all = sum_weighted_all / float(total_gt_pts_all)
    else:
        mWCov_all = 0.0
    mPrec_all = TP_all / float(P_all) if P_all > 0 else 0.0
    mRec_all  = TP_all / float(G_all) if G_all > 0 else 0.0

    with open(txt_path, "w") as f:
        f.write("====== DATASET METRICS (paired folders) ======\n")
        f.write(f"files paired: {len(pairs)}\n")
        f.write(f"IoU threshold: {iou_thresh}\n")
        if ignore_label is not None:
            f.write(f"ignore_label (GT filtered): {ignore_label}\n")
        f.write("\n")
        f.write(f"Global mCov : {mCov_all:.6f}\n")
        f.write(f"Global mWCov: {mWCov_all:.6f}\n")
        f.write(f"Global Prec : {mPrec_all:.6f}\n")
        f.write(f"Global Rec  : {mRec_all:.6f}\n")
        f.write(f"TP={TP_all}  G={G_all}  P={P_all}\n")

    print("\n====== DATASET METRICS (paired folders) ======")
    print(f"files paired: {len(pairs)}")
    print(f"IoU thr     : {iou_thresh}")
    if ignore_label is not None:
        print(f"ignore_label: {ignore_label}")
    print(f"Global mCov : {mCov_all:.6f}")
    print(f"Global mWCov: {mWCov_all:.6f}")
    print(f"Global Prec : {mPrec_all:.6f}")
    print(f"Global Rec  : {mRec_all:.6f}")
    print(f"TP={TP_all}  G={G_all}  P={P_all}")
    print(f"\nLogs saved to:\n  {csv_path}\n  {txt_path}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate instance metrics by pairing pred_dir and gt_dir")
    ap.add_argument("--pred_dir", default="/mnt/data1/split_data_instance/potato/leaf/init/renumber/potato_instance_result/cluster", help="预测标签所在目录（x y z pred_label）")
    ap.add_argument("--gt_dir",   default="/mnt/data1/split_data_instance/potato/leaf/init/renumber/potato_instance_result/Labeled", help="人工标签所在目录（x y z gt_label）")
    ap.add_argument("--output_dir", default="/mnt/data1/new_work/PVDST-main/instance/log_eval/Potato3", help="日志输出目录（默认 ./log_eval_pair）")
    ap.add_argument("--pattern", default="*.txt", help="文件通配（默认 *.txt）")
    ap.add_argument("--iou_thresh", type=float, default=0.5, help="实例匹配 IoU 阈值（默认 0.5）")
    ap.add_argument("--ignore_label", type=int, default=None,
                    help="可选：忽略的 GT 标签（仅对 GT 过滤，如茎秆=1）")
    ap.add_argument("--check_xyz", type=float,default=True,
                    help="是否检查两侧坐标逐行一致（默认不检查，仅按行对齐标签）")
    ap.add_argument("--xyz_tol", type=float, default=1e-6, help="坐标一致性容差（配合 --check_xyz 使用）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_two_folders(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        iou_thresh=args.iou_thresh,
        ignore_label=args.ignore_label,
        pattern=args.pattern,
        check_xyz=args.check_xyz,
        xyz_tol=args.xyz_tol
    )