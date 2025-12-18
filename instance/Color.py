#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量为实例分割点云赋色（最多 60 种互异颜色）。
输入：文件夹内 .txt（至少4列：X Y Z Label；Label列可用 --label_col 指定）
输出：output_dir/{原名}_colored.txt  格式：X Y Z R G B Label
同时输出：output_dir/color_legend.csv
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# -------- 经典 SCI 配色（优先使用）--------
BREWER_SET1   = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628","#F781BF","#999999"]
BREWER_SET2   = ["#66C2A5","#FC8D62","#8DA0CB","#E78AC3","#A6D854","#FFD92F","#E5C494","#B3B3B3"]
BREWER_SET3   = ["#8DD3C7","#FFFFB3","#BEBADA","#FB8072","#80B1D3","#FDB462","#B3DE69","#FCCDE5","#D9D9D9","#BC80BD","#CCEBC5","#FFED6F"]
BREWER_PAIRED = ["#A6CEE3","#1F78B4","#B2DF8A","#33A02C","#FB9A99","#E31A1C","#FDBF6F","#FF7F00","#CAB2D6","#6A3D9A","#FFFF99","#B15928"]
TABLEAU_10    = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

BASE_PALETTE = []
for lst in (TABLEAU_10, BREWER_SET1, BREWER_SET2, BREWER_SET3, BREWER_PAIRED):
    for h in lst:
        if h not in BASE_PALETTE:
            BASE_PALETTE.append(h)

def hex_to_rgb255(hex_code: str) -> Tuple[int,int,int]:
    h = hex_code.strip().lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def hsv_to_rgb255(h: float, s: float, v: float) -> Tuple[int,int,int]:
    i = int(h*6.0)
    f = h*6.0 - i
    p = int(255*(v*(1.0-s)))
    q = int(255*(v*(1.0-f*s)))
    t = int(255*(v*(1.0-(1.0-f)*s)))
    v255 = int(255*v)
    i = i % 6
    if i == 0: r,g,b = v255, t, p
    elif i == 1: r,g,b = q, v255, p
    elif i == 2: r,g,b = p, v255, t
    elif i == 3: r,g,b = p, q, v255
    elif i == 4: r,g,b = t, p, v255
    else:        r,g,b = v255, p, q
    return r,g,b

def generate_hsv_colors(n: int, seed_shift: float = 0.0) -> List[Tuple[int,int,int]]:
    """
    用黄金角在 HSV 均匀取色，保证高区分度；s,v 偏高，适合论文白底。
    交替两档明度，进一步拉开相邻颜色差异。
    """
    colors = []
    golden = 0.61803398875
    h = (0.07 + seed_shift) % 1.0
    for k in range(n):
        h = (h + golden) % 1.0
        s = 0.78
        v = 0.96 if (k % 2 == 0) else 0.84
        colors.append(hsv_to_rgb255(h, s, v))
    return colors

def build_palette_max60(n: int, max_colors: int = 60) -> List[Tuple[int,int,int,str]]:
    """
    返回长度为 min(n, max_colors) 的调色板：[(R,G,B,HEX), ...]
    先用经典配色，不够再用 HSV 取色补足，最多 60 种。
    """
    n_use = min(n, max_colors)
    out: List[Tuple[int,int,int,str]] = []
    # 先放经典色
    for hexc in BASE_PALETTE[:min(n_use, len(BASE_PALETTE))]:
        r,g,b = hex_to_rgb255(hexc)
        out.append((r,g,b,hexc))
    # 不够则补
    if n_use > len(BASE_PALETTE):
        need = n_use - len(BASE_PALETTE)
        extras = generate_hsv_colors(need, seed_shift=0.11)
        for r,g,b in extras:
            hexc = f"#{r:02X}{g:02X}{b:02X}"
            out.append((r,g,b,hexc))
    return out

def colorize_instances(xyz: np.ndarray, labels: np.ndarray, max_colors: int = 60) -> Tuple[np.ndarray, Dict[int,Tuple[int,int,int,str]]]:
    uniq = sorted(list(set(int(x) for x in labels)))
    palette = build_palette_max60(len(uniq), max_colors=max_colors)
    # 若标签数量超过 max_colors，进行映射环绕，但先提示
    if len(uniq) > max_colors:
        print(f"[WARN] 标签数 {len(uniq)} 超过 {max_colors}，颜色将循环复用，建议拆分或分层显示。")
    lab2color: Dict[int,Tuple[int,int,int,str]] = {}
    for i, lab in enumerate(uniq):
        idx = i % len(palette)
        lab2color[lab] = palette[idx]
    rgb = np.array([lab2color[int(l)][:3] for l in labels], dtype=np.int32)
    # 输出顺序：X Y Z R G B Label
    colored = np.column_stack([xyz, rgb, labels.astype(int)])
    return colored, lab2color

def save_colored_txt(path: Path, colored: np.ndarray):
    np.savetxt(path, colored, fmt="%.6f %.6f %.6f %d %d %d %d")

def save_legend_csv(path: Path, legend: Dict[int,Tuple[int,int,int,str]]):
    with path.open("w", encoding="utf-8") as f:
        f.write("Label,R,G,B,HEX\n")
        for lab in sorted(legend.keys()):
            r,g,b,hx = legend[lab]
            f.write(f"{lab},{r},{g},{b},{hx}\n")

def main():
    ap = argparse.ArgumentParser(description="批量为点云实例赋色（最多60种颜色），输出 X Y Z R G B Label")
    ap.add_argument("--input_dir", default="/mnt/data1/split_data_instance/Corn/corn01/select/init/renumber/corn_instance_result/labeled", help="输入文件夹（包含 .txt）")
    ap.add_argument("--output_dir", default="/mnt/data1/split_data_instance/Corn/corn01/select/init/renumber/corn_instance_result/labeled_color", help="输出文件夹")
    ap.add_argument("--label_col", type=int, default=-1, help="标签列索引（默认3；-1 表示最后一列）")
    ap.add_argument("--max_colors", type=int, default=60, help="最多颜色数（默认60）")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    legend_all: Dict[int,Tuple[int,int,int,str]] = {}

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        print(f"[INFO] 未在 {in_dir} 找到 .txt 文件")
        return

    for f in files:
        data = np.loadtxt(f, dtype=float)
        if data.ndim != 2 or data.shape[1] < 4:
            print(f"[WARN] {f.name} 列数不足（需≥4列），已跳过。")
            continue
        xyz = data[:, :3]
        col = args.label_col #if args.label_col >= 0 else (data.shape[1]-1)
        labels = data[:, col].astype(int)

        colored, legend = colorize_instances(xyz, labels, max_colors=args.max_colors)
        # 合并全局图例（同 label 多文件使用同色；若冲突，后者覆盖前者——通常一致）
        for k, v in legend.items():
            legend_all[k] = v

        out_path = out_dir / f"{f.stem}_colored.txt"
        save_colored_txt(out_path, colored)
        print(f"[OK] {f.name} -> {out_path.name}  （标签数：{len(set(labels.tolist()))}）")

    # 输出汇总图例
    save_legend_csv(out_dir / "color_legend.csv", legend_all)
    print(f"[OK] 颜色图例已保存：{out_dir/'color_legend.csv'}")

if __name__ == "__main__":
    main()