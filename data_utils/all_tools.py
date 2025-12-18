import os
import random

import numpy as np
import pandas as pd
import open3d as o3d
import plyfile
from plyfile import PlyElement, PlyData
import pyransac3d as pyrsc

from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, estimate_bandwidth, MeanShift, Birch, \
    AffinityPropagation, DBSCAN


def read_ply2np(path):
    """

    Args:
        path: 路径

    Returns: 放回numpy矩阵的点云

    """
    ply_read = PlyData.read(path)
    name = [ply_read["vertex"].properties[i].name for i in range(len(ply_read["vertex"].properties))]
    data = np.array(ply_read["vertex"][name[0]]).reshape(-1, 1)
    for i, name in enumerate(name[1:]):
        temp_i = np.array(ply_read["vertex"][name]).reshape(-1, 1)
        data = np.concatenate([data, temp_i], axis=1)
    return data



def SOR(np_input, filter_num=1, nb_neighbors=6, std_ratio=3):
    """

    Args:
        np_input:
        filter_num: 过滤次数
        nb_neighbors:
        std_ratio:

    Returns:

    """
    if filter_num == 0:
        return np_input
    elif filter_num > 0:
        for i in range(filter_num):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_input[:, 0:3])
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)  # 统计滤波 6  3
            if i < (filter_num - 1):
                np_input = np_input[ind]
            elif i == filter_num - 1:
                np_out = np_input[ind]
        return np_out


def land_rota(np_input):
    """

    Args:
        np_input: 输入numpy点云

    Returns: 返回拟合地面—旋转-平移后的点

    """
    label_3 = np_input[np_input[:, 6] == 3]  # 地面点
    road_pcd = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(label_3[:, 0:3])
    plane_model, inliers = road_pcd.segment_plane(distance_threshold=0.3,
                                                  ransac_n=10,
                                                  num_iterations=100)
    normal = plane_model[0:3]
    T = np.eye(4)
    R = pyrsc.get_rotationMatrix_from_vectors(normal, [0, 0, -1])
    T[:3, :3] = R
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_input[:, 0:3])
    pcd_r = pcd.transform(T)
    np_input[:, :3] = np.asarray(pcd_r.points)

    # 第二次拟合，用来平移地面平面到 z=0
    road_pcd.points = o3d.utility.Vector3dVector(np_input[np_input[:, 6] == 3][:, 0:3])
    plane_model, inliers = road_pcd.segment_plane(distance_threshold=0.3,
                                                  ransac_n=10,
                                                  num_iterations=100)

    np_input[:, 0:3] = np.asarray(pcd_r.points)
    np_input[:, 2] = np_input[:, 2] + (plane_model[3] / plane_model[2])
    np_input[:, 0:2] = np_input[:, 0:2] - np.min(np_input[:, 0:2], axis=0)  # c(z-d/c)+d
    # 只保留xyzrgb，无label
    # np_new = np_input[:, 0:6]
    np_new = np_input
    return np_new


#rota方向调整
def land_rota2(np_input):
    """

    Args:
        np_input: 输入numpy点云

    Returns: 返回拟合地面—旋转-平移后的点

    """
    label_3 = np_input[np_input[:, 6] == 3]  # 地面点
    road_pcd = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(label_3[:, 0:3])
    plane_model, inliers = road_pcd.segment_plane(distance_threshold=0.3,
                                                  ransac_n=10,
                                                  num_iterations=100)
    normal = plane_model[0:3]
    T = np.eye(4)
    R = pyrsc.get_rotationMatrix_from_vectors(normal, [0, 0, 1])
    T[:3, :3] = R
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_input[:, 0:3])
    pcd_r = pcd.transform(T)
    np_input[:, :3] = np.asarray(pcd_r.points)

    # 第二次拟合，用来平移地面平面到 z=0
    road_pcd.points = o3d.utility.Vector3dVector(np_input[np_input[:, 6] == 3][:, 0:3])
    plane_model, inliers = road_pcd.segment_plane(distance_threshold=0.3,
                                                  ransac_n=10,
                                                  num_iterations=100)

    np_input[:, 0:3] = np.asarray(pcd_r.points)
    np_input[:, 2] = np_input[:, 2] + (plane_model[3] / plane_model[2])
    np_input[:, 0:2] = np_input[:, 0:2] - np.min(np_input[:, 0:2], axis=0)  # c(z-d/c)+d
    # 只保留xyzrgb，无label
    # np_new = np_input[:, 0:6]
    np_new = np_input
    return np_new


def save_ply_from_np(np_input, ply_path):
    """

    Args:
        np_input: 输入numpy点云，需要大于6维
        ply_path: 保存路径

    Returns: 无返回

    """
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'int16'), ('green', 'int16'),
                  ('blue', 'int16')]
    points = [tuple(x) for x in np_input.tolist()]
    if np_input.shape[1] == 6:
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)
    elif np_input.shape[1] <= 6:
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if np_input.shape[1] >= 4:
            for i in range(np_input.shape[1] - 3):
                dtype_list.append((f'scalar_sf{i}', 'f4'))
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)
    elif np_input.shape[1] == 7:
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'int16'), ('green', 'int16'),
                      ('blue', 'int16'), ('scalar_sf', 'f4')]
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)
    elif np_input.shape[1] > 7:
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'int16'), ('green', 'int16'),
                      ('blue', 'int16'), ('scalar_sf', 'f4')]
        for i in range(np_input.shape[1] - 7):
            dtype_list.append((f'scalar_sf{i + 1}', 'f4'))
        vertex = np.array(points, dtype=dtype_list)
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el]).write(ply_path)


# RGB filter
def filter_rgb(pcd_input, flag=False):
    """

    Args:
        pcd_input: cloud points
        flag: 时期是否是10月以后的，确定是否需要去除黄点3

    Returns:

    """
    # 黄点去除1
    pcd_input = pcd_input[~(
            (pcd_input[:, 3] > 160) & (pcd_input[:, 5] > 160) & (pcd_input[:, 5] < 200) & (
            (pcd_input[:, 4] - pcd_input[:, 5]) > 2))]
    # 黄点去除2
    pcd_input = pcd_input[~(np.logical_and(pcd_input[:, 3] < 120, pcd_input[:, 4] > 200, pcd_input[:, 5] > 200) & (
            (pcd_input[:, 4] - pcd_input[:, 5]) > 2))]
    # 黄点去除3，用于后期叶片去黄点
    if flag == True:
        pcd_input = pcd_input[~(np.logical_and(pcd_input[:, 3] > 200, pcd_input[:, 4] > 200, pcd_input[:, 5] > 200))]
    pcd_input = pcd_input[~(pcd_input[:, 5] - pcd_input[:, 3] > 80)]  # 蓝-红 差值>80
    pcd_input = pcd_input[~np.logical_and(pcd_input[:, 3] < 50, pcd_input[:, 4] < 50, pcd_input[:, 5] < 50)]  # 删除黑点

    return pcd_input


# sacle 缩放
def ply_scale(pcd_input, pd_scale, sq_name):
    """

    Args:
        pcd_input:
        pd_scale: 缩放tabel
        sq_name: sql name

    Returns:

    """
    scale = pd_scale[pd_scale["name"] == sq_name]["scale"].values[0]
    pcd_input[:, 0:3] = pcd_input[:, 0:3] * scale
    return pcd_input




# 将地面拟合后进行旋转，平移
def run_ply_rota(file_path):
    # file_path = "../data/cabbage_set/label_help/重建点云/1012"
    files = os.listdir(file_path)
    print(files)
    for idx, file in enumerate(files):
        if ".ply" not in file:
            continue
        print(file)
        if ".pickle" in file:  # 跳过该pickle
            continue
        path = os.path.join(file_path, file)
        data_read = read_ply2np(path)
        data_af = SOR(data_read, filter_num=2)
        data_arota = land_rota(data_af)

        if "concat_" in file:
            file = file.replace("concat_", "")
        save_path = os.path.join(file_path, file.replace(".ply", "_rota.ply"))
        save_ply_from_np(data_arota, save_path)


# 将每个label点分开
def run_split(file_path="/media/zhu/HUB4T/Stratified_Transformer/cabbage_set/label_help/split", cls=5):
    files = os.listdir(file_path)
    print(files)
    for idx, file in enumerate(files):
        if ".ply" not in file:
            continue
        print(file)
        if ".pickle" in file:  # 跳过该pickle
            continue
        # 是否存在文件夹
        save_path = file_path + "/" + file.replace(".ply", "")
        if "concat_" in save_path:
            save_path = save_path.replace("concat_", "")
        elif "_rota" in save_path:
            save_path = save_path.replace("_rota", "")

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            continue

        path = os.path.join(file_path, file)
        data_read = read_ply2np(path)
        # 5是cls 个数
        for i in range(5):
            locals()[f"label{i}"] = data_read[data_read[:, 6] == i]
        for i in range(5):
            save_ply_from_np(locals()[f"label{i}"], save_path + f"/label{i}.ply")


# 将每个label点分开,以及颜色滤波
def run_split_filter(file_path="../data/cabbage_set/label_help/test_save/p3/901", cls=5):
    # file_path = "../data/cabbage_set/label_help/test_save/p3/901"
    files = os.listdir(file_path)
    print(files)
    for idx, file in enumerate(files):
        if ".ply" not in file:
            continue
        print(file)
        if ".pickle" in file:  # 跳过该pickle
            continue
        # 是否存在文件夹
        save_path = file_path + "/" + file.replace(".ply", "")
        if "concat_" in save_path:
            save_path = save_path.replace("concat_", "")
        if "_rota" in save_path:
            save_path = save_path.replace("_rota", "")

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            continue

        path = os.path.join(file_path, file)
        data_read = read_ply2np(path)
        data_read = SOR(data_read, filter_num=2)
        # 5是cls 个数
        for i in range(5):
            locals()[f"label{i}"] = data_read[data_read[:, 6] == i]
            # 1 2 进行 rgb filter
            if i == 1 or i == 2:
                locals()[f"label{i}"] = filter_rgb(locals()[f"label{i}"])
            locals()[f"label{i}"] = SOR(locals()[f"label{i}"], filter_num=1)
        # save
        for i in range(5):
            save_ply_from_np(locals()[f"label{i}"], save_path + f"/label{i}.ply")


def run_filter_scale(file_path):
    files = os.listdir(file_path)
    print(files)

    PCD_measure = pd.read_excel("PCD_measurement.xlsx", header=None, names=["name", "scale"])
    for idx, file in enumerate(files):
        if ".ply" not in file:
            continue
        print(file)
        if ".pickle" in file:  # 跳过该pickle
            continue

    # PCD_measure[PCD_measure["name"]=="901-02"]["scale"].values[0]
    files = os.listdir(file_path)

    file_name = "concat_1012-09_rota-sample200w.ply"
    ply_np = read_ply2np(file_name)
    if "concat_" in file_name:
        file_name = file_name.replace("concat_", "")
    if "_rota" in file_name:
        file_name = file_name.replace("_rota", "")
    sq_name = file_name.split("-")[0:2]
    sq_name = sq_name[0] + "-" + sq_name[1]
    print(sq_name)
    scale = PCD_measure[PCD_measure["name"] == sq_name]["scale"].values[0]
    print(scale)
    ply_np[:, 0:3] = ply_np[:, 0:3] * scale
    save_ply_from_np(ply_np, file_name)


# filter scale rota集合
def run_filter_scale_rota(file_path="../data/cabbage_set/label_help/train"):
    path = file_path
    save_path = "../data/cabbage_set/label_help/train_afterfilter"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    PCD_measure = pd.read_excel("PCD_measurement.xlsx", header=None, names=["name", "scale"])
    for root, dirs, files in os.walk(path):
        for file in files:
            # 获取文件路径
            print(f"读取到路径{os.path.join(root, file)}")
            # 读取文件
            if ".ply" not in file:
                continue
            read_np = read_ply2np(os.path.join(root, file))
            # 创建save文件
            if not os.path.exists(root.replace("train", "train_afterfilter")):
                os.mkdir(root.replace("train", "train_afterfilter"))
            if "concat_" in file:
                file = file.replace("concat_", "")
            if "_rota" in file:
                file = file.replace("_rota", "")
            sq_name = file.split("-")[0:2]
            sq_name = sq_name[0] + "-" + sq_name[1]
            if ".ply" in sq_name:
                sq_name = sq_name.replace(".ply", "")
            print(f"sq_name: {sq_name}")

            # scale_value = PCD_measure[PCD_measure["name"] == sq_name]["scale"].values[0]  # 缩放scale
            # 对点云进行操作-----------------------------------
            ply_np = SOR(read_np, filter_num=3, nb_neighbors=6, std_ratio=3)  # filter_num=0没有滤波
            ply_np = land_rota2(ply_np) #rota2与rota1的方向不同
            ply_np = filter_rgb(ply_np)
            # ply_np = ply_scale(ply_np, PCD_measure, sq_name)
            ply_np = SOR(ply_np, filter_num=2, nb_neighbors=6, std_ratio=3)  # filter_num=0没有滤波

            save_name = os.path.join(root.replace("train", "train_afterfilter"), file)
            print(f"保存：{save_name}")
            save_ply_from_np(ply_np, save_name)
            # print(root)
            # 获取文件路径
            # print(os.path.join(root, file))


# 随机下采样
def run_random_sample():
    filepath = "maize_label/maize_raw"
    savepath = "maize_label/maize_100W"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    #PCD_measure = pd.read_excel("PCD_measurement.xlsx", header=None, names=["name", "scale"])
    for file in os.listdir(filepath):
        print(f"读取：{file}")
        # if ".ply" not in file:
        #     continue
        #ply_np = read_ply2np(os.path.join(filepath, file))
        ply_np = np.loadtxt(os.path.join(filepath, file))

        if "_test" in file:
            file = file.replace("_test", "")
        if "_rota" in file:
            file = file.replace("_rota", "")

        # 对点云进行操作
        # ply_np = SOR(ply_np, filter_num=1, nb_neighbors=6, std_ratio=3)  # filter_num=0没有滤波
        # ply_np = filter_rgb(ply_np)
        # ply_np = SOR(ply_np, filter_num=1, nb_neighbors=6, std_ratio=3)  # filter_num=0没有滤波
        if ply_np.shape[0] > 1000000:
            sample_idx = np.random.choice(len(ply_np), 1000000, replace=False)  # s随机采500个数据，这种随机方式也可以自己定义
            ply_np_sample = ply_np[sample_idx]
        else:
            ply_np_sample = ply_np

        #save_ply_from_np(ply_np_sample, os.path.join(savepath, file.replace(".ply", "_sample10w.ply")))
        np.savetxt(os.path.join(savepath, file),ply_np_sample)

# run scale
def run_scale():
    filepath = "../data/cabbage_set/label_help/train"
    savepath = "../data/cabbage_set/label_help/train_sample"
    PCD_measure = pd.read_excel("PCD_measurement.xlsx", header=None, names=["name", "scale"])
    for file in os.listdir(filepath):
        print(f"读取：{file}")
        if ".ply" not in file:
            continue
        ply_np = read_ply2np(os.path.join(filepath, file))
        if "_test" in file:
            file = file.replace("_test", "")
        if "_rota" in file:
            file = file.replace("_rota", "")

        sql_name = file.split(".")[0]
        ply_np = ply_scale(ply_np, pd_scale=PCD_measure, sq_name=sql_name)

        save_ply_from_np(ply_np, os.path.join(savepath, file))


#预处理后，将文件中的ply点云叶子部分提取，下采样，聚类，每个叶片存储到文件中
def run_filter_split_cluster(file_path="../data/cabbage_set/label_help/train"):
    save_path = "../data/cabbage_set/label_help/leaf_set"
    # file_path = "../data/cabbage_set/label_help/test_save/p3/901"
    files = os.listdir(file_path)
    print(files)
    for idx, file in enumerate(files):
        if ".ply" not in file:
            continue
        print(file)
        if ".pickle" in file:  # 跳过该pickle
            continue
        # 是否存在文件夹
        save_path2 = save_path + "/" + file.replace(".ply", "")
        if "concat_" in save_path2:
            save_path2 = save_path2.replace("concat_", "")
        if "_rota" in save_path2:
            save_path2 = save_path2.replace("_rota", "")
        if "_test" in save_path2:
            save_path2 = save_path2.replace("_test", "")

        if not os.path.exists(save_path2):
            os.mkdir(save_path2)
        # else:
        #     continue

        path = os.path.join(file_path, file)
        data_read = read_ply2np(path)
        # 5是cls 个数
        data_read = SOR(data_read,filter_num=2, nb_neighbors=6, std_ratio=2)
        for i in range(5):
            locals()[f"label{i}"] = data_read[data_read[:, 6] == i]
            # 1 2 进行 rgb filter
            if i == 1 or i == 2:
                locals()[f"label{i}"] = filter_rgb(locals()[f"label{i}"])
                locals()[f"label{i}"] = SOR(locals()[f"label{i}"], filter_num=2, nb_neighbors=6, std_ratio=2)
        locals()[f"label{i}"] = SOR(locals()[f"label{i}"], filter_num=1)
        #存储滤波后的点云
        # save_ply_from_np(data_read, save_path + "/" + file)
        # 聚类
        leaf = locals()[f"label{2}"]
        sample_idx = np.random.choice(len(leaf), 50000, replace=False)  # s随机采500个数据，这种随机方式也可以自己定义
        leaf = leaf[sample_idx]
        leaf = leaf[:, :6]
        #存储叶片点云
        save_ply_from_np(leaf, save_path+"/"+file)

        dataset_X = leaf[:, 0:3]
        # result = OPTICS(min_samples=15, max_eps=0.0015).fit(dataset_X)
        result = DBSCAN(min_samples=10, eps=0.005).fit(dataset_X)
        labels = result.labels_
        print(f"labels_num:{len(set(labels))}")
        #------------open3d dbscan--------------------
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(leaf[:, 0:3])
        # labels = pcd.cluster_dbscan(0.005, 10, print_progress=True)
        new = np.concatenate([leaf[:, 0:7], np.asarray(labels).reshape(-1, 1)], axis=1)

        for i in set(labels):
            if len(new[new[:, 6] == i]) > 5000:
                save_ply_from_np(new[new[:, 6] == i], save_path2 + f"/leaf{i}.ply")

        # for i in [2]:
        #     save_ply_from_np(locals()[f"label{i}"], save_path + f"/label{i}.ply")

#把过滤rgb，滤波后的点云中的叶片分割出来
def run_split_leaf(file_path = "/media/zhu/HUB4T/Stratified_Transformer/cabbage_set/label_help/rota_filter_scale_RGBF"):
    path = file_path
    save_path = "/media/zhu/HUB4T/Stratified_Transformer/cabbage_set/label_help/leaf_set"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    PCD_measure = pd.read_excel("PCD_measurement.xlsx", header=None, names=["name", "scale"])
    for root, dirs, files in os.walk(path):
        for file in files:
            # 获取文件路径
            print(f"读取到路径{os.path.join(root, file)}")
            # 读取文件
            if ".ply" not in file:
                continue
            read_np = read_ply2np(os.path.join(root, file))
            # 创建save文件
            if not os.path.exists(root.replace("rota_filter_scale_RGBF", "leaf_set")):
                os.mkdir(root.replace("rota_filter_scale_RGBF", "leaf_set"))
            if "concat_" in file:
                file = file.replace("concat_", "")
            if "_rota" in file:
                file = file.replace("_rota", "")
            sq_name = file.split("-")[0:2]
            sq_name = sq_name[0] + "-" + sq_name[1]
            if ".ply" in sq_name:
                sq_name = sq_name.replace(".ply", "")
            print(f"sq_name: {sq_name}")

            leaf = read_np[read_np[:, 6]==2]
            np.random.seed(1)   #设置随机种子
            sample_idx = np.random.choice(len(leaf), 50000, replace=False)  # s随机采500个数据，这种随机方式也可以自己定义
            leaf = leaf[sample_idx]

            if file.split("-")[0][0] == "9":
                min_samples = 10
                eps = 0.005
            elif file.split("-")[0][0:2] == "10":
                min_samples = 10
                eps = 0.01
            else:
                min_samples = 10
                eps = 0.01
            #聚类
            dataset_X = leaf[:, 0:3]
            result = DBSCAN(min_samples=min_samples, eps=eps).fit(dataset_X)
            labels = result.labels_
            print(f"labels_num:{len(set(labels))}")
            new = np.concatenate([leaf[:, 0:7], np.asarray(labels).reshape(-1, 1)], axis=1)

            #保存leaf
            save_ply_from_np(new, os.path.join(root.replace("rota_filter_scale_RGBF", "leaf_set"), file))
            #聚类label拆分
            save_path2 = os.path.join(root.replace("rota_filter_scale_RGBF", "leaf_set"), file.replace(".ply", ""))
            if not os.path.exists(save_path2):
                os.mkdir(save_path2)

            for i in set(labels):
                if len(new[new[:, 7] == i]) > 4000:
                    # np.savetxt(save_path2 + f"/leaf{i}.txt", new[new[:, 7] == i],delimiter=",", fmt="%s")
                    save_ply_from_np(new[new[:, 7] == i], save_path2 + f"/leaf{i}.ply")

            # save_name = os.path.join(root.replace("rota_filter_scale_RGBF", "leaf_set"), file)
            # print(f"保存：{save_name}")
            # save_ply_from_np(ply_np, save_name)
            # print(root)
            # 获取文件路径
            # print(os.path.join(root, file))


if __name__ == '__main__':
    # run_ply_rota()
    # run_split()
    # run_split_filter(file_path=path + list[i])
    # list = ["901", "907", "914"]
    # run_filter_scale_rota()
    # run_split_leaf()
    run_random_sample()
    # run_scale()
    # run_filter_split_cluster()
    """*----------------------------------*"""
    # list = ["1123","1205"]
    # path = "/media/zhu/HUB4T/Stratified_Transformer/cabbage_set/label_help/重建点云/"
    # for i in range(len(list)):
    #     run_ply_rota(file_path=path + list[i])
