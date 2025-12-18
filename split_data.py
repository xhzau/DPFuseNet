import os
import numpy as np

def split_point_cloud_by_label(input_folder, output_folder):
    # 获取文件夹中的所有 .txt 文件
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for file in files:
        file_path = os.path.join(input_folder, file)
        # 读取点云数据
        data = np.loadtxt(file_path)
        # 假设标签在最后一列
        labels = data[:, -1]
        xyz = data[:,0:3]
        points = np.concatenate((xyz, labels.reshape(-1,1)), axis=1)
        # 获取唯一标签
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # 获取当前标签的点
            subset = points[labels == label]
            # 创建标签文件夹
            label_folder = os.path.join(output_folder, f"label_{int(label)}")
            os.makedirs(label_folder, exist_ok=True)
            # 构建新的文件名
            output_file = f"{os.path.splitext(file)[0]}_label_{int(label)}.txt"
            output_path = os.path.join(label_folder, output_file)
            # 保存到新的 .txt 文件
            np.savetxt(output_path, subset, fmt='%.6f')
            print(f"Saved {output_path}")

input_folder = '/mnt/data1/new_work/PVDST-main/log/part_seg/loop=10/1028_Abliation/Potato_1205/test/output/pred_label/032215-'
output_folder = '/mnt/data1/new_work/PVDST-main/log/part_seg/loop=10/1028_Abliation/Potato_1205/test/output/pred_label/032215-/gt'
split_point_cloud_by_label(input_folder, output_folder)