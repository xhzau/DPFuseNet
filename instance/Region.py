import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_kernels
from scipy.sparse.linalg import eigsh
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples


def read_point_cloud(file_path):
    return np.loadtxt(file_path)


# 存储带标签的点云数据
def save_point_cloud_with_labels(file_path, points, labels):
    data_with_labels = np.hstack((points, labels.reshape(-1, 1)))
    np.savetxt(file_path, data_with_labels, fmt='%.6f')


# 谱聚类
def spectral_clustering(points, sigma=0.1, max_k=10):
    # 构建相似度矩阵
    # similarity_matrix = pairwise_kernels(points, metric='rbf', gamma=1 / (2 * sigma ** 2))
    similarity_matrix = construct_similarity_matrix(points, 10, sigma)

    # 计算拉普拉斯矩阵
    degree_matrix = np.diag(similarity_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    # 计算特征值和特征向量
    eigvals, eigvecs = eigsh(laplacian_matrix, k=max_k, which='SM')

    # 通过特征值间隔选择聚类数 k
    gaps = np.diff(eigvals)
    k = np.argmax(gaps) + 1  # 特征值间隔最大的位置

    # 使用前 k 个特征向量进行聚类
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(eigvecs[:, :k])

    return labels + 1  # 标签从1开始


def construct_similarity_matrix(points, n_neighbor=10, sigma=0.2):
    # 使用最近邻构造稀疏的相似度矩阵
    n_neighbors = min(n_neighbor, len(points))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    sigma = np.mean(distances[:, -1])

    # 使用高斯核函数计算相似度
    row_indices = np.repeat(np.arange(points.shape[0]), n_neighbors)
    col_indices = indices.flatten()
    weights = np.exp(-distances.flatten() ** 2 / (2 * sigma ** 2))

    similarity_matrix = csr_matrix((weights, (row_indices, col_indices)), shape=(points.shape[0], points.shape[0]))
    return similarity_matrix

def construct_similarity_matrix_aug(points, n_neighbors=16, snn_alpha=1.5):
    n_samples = len(points)
    if n_samples == 0:
        return csr_matrix((0, 0))
    n_neighbors = min(max(2, n_neighbors), n_samples)
    # kNN 搜索
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    # 自适应带宽：每个点的第 k 近邻距离
    sigmas = distances[:, -1].copy()
    sigmas[sigmas < 1e-6] = 1e-6
    neigh_sets = [set(row) for row in indices]
    row_idx, col_idx, weights = [], [], []
    for i in range(n_samples):
        for jpos, j in enumerate(indices[i]):
            if i == j:
                continue
            # 互近邻
            if i in neigh_sets[j]:
                sigma_ij = sigmas[i] * sigmas[j]
                w = np.exp(-(distances[i, jpos] ** 2) / sigma_ij)
                row_idx.append(i)
                col_idx.append(j)
                weights.append(w)
    # SNN 修正
    if snn_alpha and snn_alpha > 0.0:
        for k in range(len(weights)):
            i, j = row_idx[k], col_idx[k]
            shared = len(neigh_sets[i].intersection(neigh_sets[j]))
            snn_val = (shared / float(n_neighbors)) ** snn_alpha
            weights[k] *= snn_val

    # 构造稀疏矩阵 + 对称化
    A = csr_matrix((weights, (row_idx, col_idx)), shape=(n_samples, n_samples))
    A = 0.5 * (A + A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A



def estimate_curvature(points, n_neighbors=10):
    # 估计每个点的局部曲率
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)

    curvature = np.zeros(points.shape[0])

    for i in range(points.shape[0]):
        # 提取邻居点
        neighbors = points[indices[i]]

        # 使用PCA分析局部结构
        pca = PCA(n_components=2)
        pca.fit(neighbors)

        # 计算曲率：使用特征值比率
        eigenvalues = pca.explained_variance_
        if len(eigenvalues) > 1:
            curvature[i] = eigenvalues[1] / eigenvalues[0]
        else:
            curvature[i] = 0

    return curvature


def construct_similarity_matrix_with_curvature(points, n_neighbor=10, sigma=0.2, alpha=1.0):
    n_neighbors = min(n_neighbor, len(points))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    sigma = np.mean(distances[:, -1])

    # 计算曲率
    curvature = estimate_curvature(points, n_neighbors)

    # 使用高斯核函数计算相似度，并结合曲率信息
    row_indices = np.repeat(np.arange(points.shape[0]), n_neighbors)
    col_indices = indices.flatten()
    weights = np.exp(-distances.flatten() ** 2 / (2 * sigma ** 2))

    # 调整相似度权重，结合曲率信息
    curvature_weights = np.exp(-alpha * curvature[row_indices])
    adjusted_weights = weights * curvature_weights

    similarity_matrix = csr_matrix((adjusted_weights, (row_indices, col_indices)),
                                   shape=(points.shape[0], points.shape[0]))
    return similarity_matrix


def spectral_clustering_later(points, initial_labels, n_neighbors=16, sigma=0.1, max_k=6):
    final_labels = np.zeros_like(initial_labels)
    n_neighbors = n_neighbors
    for label in np.unique(initial_labels):
        # 选择属于当前初步标签的点
        subset = points[initial_labels == label]
        # 构建相似度矩阵
        # similarity_matrix = pairwise_kernels(subset, metric='rbf', gamma=1 / (2 * sigma ** 2))
        similarity_matrix = construct_similarity_matrix(subset, n_neighbors, sigma)
        # similarity_matrix = construct_similarity_matrix_with_curvature(subset, n_neighbors, sigma)
        # 计算拉普拉斯矩阵
        # degree_matrix = np.diag(similarity_matrix.sum(axis=1))
        degree_matrix = np.diag(similarity_matrix.sum(axis=1).A1)

        laplacian_matrix = degree_matrix - similarity_matrix

        # 计算特征值和特征向量
        eigvals, eigvecs = eigsh(laplacian_matrix, k=min(max_k, len(subset)), which='SM')

        # 通过特征值间隔选择聚类数 k
        gaps = np.diff(eigvals)
        k = np.argmax(gaps) + 1  # 特征值间隔最大的位置

        # 使用前 k 个特征向量进行聚类
        eigvecs_normalized = normalize(eigvecs[:, :k])
        kmeans = KMeans(n_clusters=k)
        # eigvecs_normalized = normalize(eigvecs[:, :min(max_k, len(subset))])
        # kmeans = KMeans(n_clusters=min(max_k, len(subset)))
        sub_labels = kmeans.fit_predict(eigvecs_normalized)

        # 更新最终标签
        final_labels[initial_labels == label] = sub_labels + label * max_k  # 确保标签不重叠

    return final_labels


def update_centroid(points, labels):
    """计算给定标签的新的质心"""
    return np.array([points[labels == label].mean(axis=0) for label in np.unique(labels) if np.any(labels == label)])


def merge_clusters(points, labels, distance_threshold):
    merged_labels = labels.copy()

    while True:
        merged_any = False  # 标记是否有合并发生
        unique_labels = np.unique(merged_labels)  # 在每次循环开始时更新唯一标签

        # 计算所有簇的质心
        centroids = update_centroid(points, merged_labels)

        for label in unique_labels:
            # 如果该簇已经合并过，跳过
            if np.any(merged_labels == label) == False:
                continue

            # 当前簇的点
            current_cluster_points = points[merged_labels == label]
            if current_cluster_points.size == 0:
                continue

            current_centroid = current_cluster_points.mean(axis=0)

            # 计算当前质心与其他簇质心之间的距离
            distances = np.linalg.norm(centroids - current_centroid, axis=1)

            # 使用布尔索引获取满足条件的索引
            close_clusters = np.unique(np.where((distances > 0) & (distances < distance_threshold))[0])  # 获取索引

            if close_clusters.size == 0:
                continue  # 如果没有更多的簇可以合并，跳过

            # 合并这些簇
            for close_label in close_clusters:
                # 只在 merged_labels 中存在时才进行合并
                if np.any(merged_labels == close_label):
                    merged_labels[merged_labels == close_label] = label
                    merged_any = True

        # 如果没有合并任何簇，退出循环
        if not merged_any:
            break

    return merged_labels



def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)

    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return centroid,m,pc


def save_segmented_results_with_labels(segments, file_path):
    with open(file_path, 'w') as f:
        for segment_id, points in segments.items():
            for point in points:
                f.write(f" {point[0]} {point[1]} {point[2]} {segment_id}\n")


def region_growing(points, distance_threshold, min_points_threshold):
    visited = np.zeros(len(points), dtype=bool)
    segments = {}
    segment_id = 0

    def grow_region(seed_index):
        to_visit = [seed_index]
        segment_points = []
        while to_visit:
            current_index = to_visit.pop()
            if not visited[current_index]:
                visited[current_index] = True
                segment_points.append(current_index)

                distances = distance.cdist([points[current_index]], points)[0]
                neighbors = np.where((distances < distance_threshold) & (~visited))[0]
                to_visit.extend(neighbors)
        return segment_points

    for i in range(len(points)):
        if not visited[i]:
            segment_points = grow_region(i)
            if segment_points:
                segments[segment_id] = segment_points
                segment_id += 1
    # 重新分配小于阈值的分段
    for seg_id, seg_points in list(segments.items()):
        if len(seg_points) < min_points_threshold:
            del segments[seg_id]
            for point_index in seg_points:
                distances = {s_id: np.min(distance.cdist([points[point_index]], points[segments[s_id]])) for s_id in
                             segments}
                closest_segment_id = min(distances, key=distances.get)
                segments[closest_segment_id].append(point_index)
    # 将点索引转为实际点
    for seg_id in segments:
        segments[seg_id] = points[segments[seg_id]]
    return segments



def compute_normals(points, k=10):
    normals = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)

    for i in range(len(points)):
        neighbors = points[indices[i]]
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normals.append(normal)

    return np.array(normals)


def compute_centroid(points):
    return np.mean(points, axis=0)


def adjust_normals(points, normals, reference_point):
    adjusted_normals = []
    for point, normal in zip(points, normals):
        vector_to_reference = reference_point - point
        if np.dot(normal, vector_to_reference) < 0:
            normal = -normal
        adjusted_normals.append(normal)
    return np.array(adjusted_normals)



def construct_similarity_vd_matrix(points, normals, n_neighbor=10, sigma=0.2, alpha=0.5):
    # 使用最近邻构造稀疏的相似度矩阵
    n_neighbors = min(n_neighbor, len(points))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    sigma = np.mean(distances[:, -1])
    # 归一化法向量
    normals_normalized = normalize(normals)
    # 使用高斯核函数计算位置相似度
    row_indices = np.repeat(np.arange(points.shape[0]), n_neighbors)
    col_indices = indices.flatten()
    spatial_weights = np.exp(-distances.flatten() ** 2 / (2 * sigma ** 2))
    # 计算法向量相似度
    normal_weights = np.array([np.exp(
        -np.arccos(np.clip(np.dot(normals_normalized[i], normals_normalized[j]), -1.0, 1.0)) ** 2 / (2 * sigma ** 2))
                               for i in range(points.shape[0]) for j in indices[i]])
    # 合并相似度
    weights = alpha * spatial_weights + (1 - alpha) * normal_weights
    similarity_matrix = csr_matrix((weights, (row_indices, col_indices)), shape=(points.shape[0], points.shape[0]))
    return similarity_matrix



def calculate_centroid_clister(points):
    return np.mean(points, axis=0)


def calculate_normal_cluster(points, viewpoint):
    # 计算每个点到视点的向量
    vectors = points - viewpoint
    # 计算协方差矩阵
    covariance_matrix = np.cov(vectors, rowvar=False)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # 法向量为特征值最小的特征向量
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    return normal


def calculate_cluster_sizes(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    return cluster_sizes



def compute_cluster_variance(subset):
    center = np.mean(subset, axis=0)
    distances = np.linalg.norm(subset - center, axis=1)
    variance = np.var(distances)
    return variance


def spectral_clustering_later_n(points, initial_labels, n_neighbors=16, sigma=0.1, max_k=6, variance_threshold=0.01):
    final_labels = np.zeros_like(initial_labels)

    for label in np.unique(initial_labels):
        subset = points[initial_labels == label]
        # 计算簇内方差
        variance = compute_cluster_variance(subset)

        # 判断是否需要进一步分割
        if variance > variance_threshold:
            similarity_matrix = construct_similarity_matrix(subset, n_neighbors, sigma)
            similarity_matrix = construct_similarity_matrix_aug(subset, n_neighbors, sigma)

            degree_matrix = np.diag(similarity_matrix.sum(axis=1).A1)
            laplacian_matrix = degree_matrix - similarity_matrix
            eigvals, eigvecs = eigsh(laplacian_matrix, k=min(max_k, len(subset)), which='SM')
            gaps = np.diff(eigvals)
            k = np.argmax(gaps) + 1
            if gaps[k - 1] > 0:
                eigvecs_normalized = normalize(eigvecs[:, :k])
                kmeans = KMeans(n_clusters=k)
                sub_labels = kmeans.fit_predict(eigvecs_normalized)
                final_labels[initial_labels == label] = sub_labels + label * max_k

            else:
                final_labels[initial_labels == label] = label * max_k
        else:
            final_labels[initial_labels == label] = label * max_k

    final_unique_labels = np.unique(final_labels)
    new_labels = {old_label: new_label for new_label, old_label in enumerate(final_unique_labels, start=2)}
    renumbered_labels = np.vectorize(new_labels.get)(final_labels)

    return renumbered_labels


def calculate_silhouette_scores(points, labels):
    # 计算每个点的轮廓系数
    silhouette_vals = silhouette_samples(points, labels)
    # 计算每个簇的平均轮廓系数
    unique_labels = np.unique(labels)
    silhouette_dict = {}
    for label in unique_labels:
        cluster_silhouette_vals = silhouette_vals[labels == label]
        silhouette_dict[label] = np.mean(cluster_silhouette_vals)
    return silhouette_dict


def compute_silhouette(points, labels):
    silhouette_vals = silhouette_samples(points, labels)
    return silhouette_vals


def spectral_clustering_later_lk(points, initial_labels, n_neighbors=16, sigma=0.1, max_k=6, s_threshold=0.5):
    final_labels = np.zeros_like(initial_labels)

    # 使用初始标签计算所有点的轮廓系数
    silhouette_vals = compute_silhouette(points, initial_labels)

    for label in np.unique(initial_labels):
        subset = points[initial_labels == label]
        subset_silhouette_vals = silhouette_vals[initial_labels == label]
        avg_silhouette = np.mean(subset_silhouette_vals)

        # 判断是否需要进一步分割
        if avg_silhouette < s_threshold:
            # similarity_matrix = construct_similarity_matrix(subset, n_neighbors, sigma)
            similarity_matrix = construct_similarity_matrix_aug(subset, n_neighbors, sigma)


            degree_matrix = np.diag(similarity_matrix.sum(axis=1).A1)
            laplacian_matrix = degree_matrix - similarity_matrix
            eigvals, eigvecs = eigsh(laplacian_matrix, k=min(max_k, len(subset)), which='SM')
            gaps = np.diff(eigvals)
            k = np.argmax(gaps) + 1

            if gaps[k - 1] > 0:
                eigvecs_normalized = normalize(eigvecs[:, :k])
                kmeans = KMeans(n_clusters=k)
                sub_labels = kmeans.fit_predict(eigvecs_normalized)
                final_labels[initial_labels == label] = sub_labels + label * max_k
            else:
                final_labels[initial_labels == label] = label * max_k
        else:
            final_labels[initial_labels == label] = label * max_k

    final_unique_labels = np.unique(final_labels)
    new_labels = {old_label: new_label for new_label, old_label in enumerate(final_unique_labels, start=2)}
    renumbered_labels = np.vectorize(new_labels.get)(final_labels)

    return renumbered_labels


def renumber_labels(points):
    labels = points[:, -1]
    unique_labels = np.unique(labels)
    new_labels = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=2)}
    renumbered_labels = np.vectorize(new_labels.get)(labels)
    points[:, -1] = renumbered_labels
    return points


def should_merge(cluster1, cluster2, distance_threshold):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    distance = np.linalg.norm(centroid1 - centroid2)
    return distance < distance_threshold




def separate_clusters(points, labels):
    unique_labels = np.unique(labels)
    clusters = [points[labels == label] for label in unique_labels]
    return clusters


def should_merge(cluster1, cluster2, distance_threshold):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    distance = np.linalg.norm(centroid1 - centroid2)
    return distance < distance_threshold


def merge_clusters(clusters, distance_threshold):
    merged_clusters = []
    used = set()

    for i, cluster1 in enumerate(clusters):
        if i in used:
            continue
        current_cluster = cluster1
        for j, cluster2 in enumerate(clusters):
            if i != j and j not in used and should_merge(current_cluster, cluster2, distance_threshold):
                current_cluster = np.vstack((current_cluster, cluster2))
                used.add(j)
        merged_clusters.append(current_cluster)
        used.add(i)

    return merged_clusters


def save_merged_clusters(merged_clusters, output_file):
    with open(output_file, 'w') as f:
        for i, cluster in enumerate(merged_clusters):
            for point in cluster:
                f.write(f"{point[0]} {point[1]} {point[2]} {i}\n")


def calculate_shared_neighbors(cluster1, cluster2, all_points, n_neighbors=5):
    """计算两个簇之间的共享邻居数量"""
    if cluster1.shape[0] == 0 or cluster2.shape[0] == 0:
        return 0
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(all_points)
    neighbors1 = neighbors.kneighbors(cluster1, return_distance=False)
    neighbors2 = neighbors.kneighbors(cluster2, return_distance=False)

    shared_count = 0
    for n1 in neighbors1:
        for n2 in neighbors2:
            shared_count += len(set(n1) & set(n2))
    return shared_count

def merge_clusters_based_on_neighbors(points,labels, n_neighbors=5, merge_threshold=10):
    """基于共享邻居数量合并簇"""
    unique_labels = np.unique(labels)
    merge_info = []

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            label1 = unique_labels[i]
            label2 = unique_labels[j]
            cluster1 = points[labels == label1]
            cluster2 = points[labels == label2]

            shared_neighbors = calculate_shared_neighbors(cluster1, cluster2, points, n_neighbors)
            merge_info.append((shared_neighbors, label1, label2))
    merge_info.sort(reverse=True, key=lambda x: x[0])
    # 合并簇
    for shared_neighbors, label1, label2 in merge_info:
        if shared_neighbors > merge_threshold:
            labels[labels == label2] = label1
            print(f"Merged Cluster {label2} into Cluster {label1}")

    return points, labels


import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict


def merge_clusters_based_on_mhd(file_path, threshold):
    # 读取点云数据
    data = np.loadtxt(file_path)
    points = data[:, :3]
    labels = data[:, -1].astype(int)

    # 获取唯一标签
    unique_labels = np.unique(labels)
    label_map = {label: label for label in unique_labels}

    # 初始化一个字典来存储簇的点
    clusters = {label: points[labels == label] for label in unique_labels}

    # 合并簇
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                # 计算簇i和簇j之间的曼哈顿距离
                distances = cdist(clusters[label_i], clusters[label_j], metric='cityblock')
                average_distance = np.mean(distances)

                # 如果距离小于阈值，合并簇
                if average_distance < threshold:
                    # 将簇j合并到簇i中
                    clusters[label_i] = np.vstack((clusters[label_i], clusters[label_j]))
                    label_map[label_j] = label_i

    # 更新标签以反映合并结果
    merged_labels = labels.copy()
    for original_label in unique_labels:
        merged_labels[labels == original_label] = label_map[original_label]

    # 打印合并后的结果
    unique_merged_labels = np.unique(merged_labels)
    print(f"合并后的簇标签: {unique_merged_labels}")
    return points,merged_labels

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def construct_similarity_matrix_enhanced(points,
                                         k_neighbor=16,
                                         use_normal=True,
                                         normal_k=25,
                                         snn_alpha=1.5,
                                         tau=0.2):

    n = len(points)
    if n <= 1:
        return csr_matrix((n, n))

    # === Step 1: kNN 查询 ===
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbor, n)).fit(points)
    dists, idxs = nbrs.kneighbors(points)

    # 自适应带宽：σ_i = 第 k 近邻距离
    sigmas = dists[:, -1]
    sigmas[sigmas < 1e-6] = 1e-6

    # === Step 2: 互近邻 + 高斯权重 ===
    row_idx, col_idx, weights = [], [], []
    for i in range(n):
        for j_pos, j in enumerate(idxs[i]):
            if i == j:
                continue
            # 互近邻判断
            if i in idxs[j]:
                sigma_ij = sigmas[i] * sigmas[j]
                w = np.exp(- (dists[i, j_pos]**2) / sigma_ij)
                row_idx.append(i)
                col_idx.append(j)
                weights.append(w)

    # === Step 3: 共享近邻（SNN） ===
    # 构建邻居集合
    neigh_sets = [set(idxs[i]) for i in range(n)]
    for k in range(len(weights)):
        i, j = row_idx[k], col_idx[k]
        shared = len(neigh_sets[i].intersection(neigh_sets[j]))
        snn_val = (shared / float(k_neighbor)) ** snn_alpha
        weights[k] *= snn_val

    # === Step 4: 法向角度项（可选） ===
    if use_normal:
        # 局部 PCA 求法向
        normals = np.zeros_like(points)
        nn_n = NearestNeighbors(n_neighbors=min(normal_k, n)).fit(points)
        _, idxs_n = nn_n.kneighbors(points)
        for i in range(n):
            pts_nb = points[idxs_n[i]]
            cov = np.cov((pts_nb - pts_nb.mean(axis=0)).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, np.argmin(eigvals)]  # 最小特征值对应法向
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

        for k in range(len(weights)):
            i, j = row_idx[k], col_idx[k]
            cos_val = abs(np.dot(normals[i], normals[j]))
            angle_weight = np.exp(-(1 - cos_val) / tau)
            weights[k] *= angle_weight

    # === Step 5: 构造稀疏矩阵并对称化 ===
    W = csr_matrix((weights, (row_idx, col_idx)), shape=(n, n))
    W = 0.5 * (W + W.T)
    W.setdiag(0)
    W.eliminate_zeros()

    return W


# 主程序
# 示例使用
# input_file = './maize_2/032210-21_label_2.txt'
# input_file = './Maize-miao/1.0/032230-29_1.0.txt'
#
# input_file ='./tomato-ph4d/032211-9 - Cloud.txt'
input_file = '/mnt/data1/split_data_instance/Cabbage/rm_4-3/down/split/leaf/mvs_1116_07.txt'
# input_file = '/home/zero/mnt/sda6/new_work/syau_single_maize/leaf/LD145-10-6-1.txt'
# input_file = '/home/zero/mnt/sda6/split_data_instance/Corn/leaf/032213-67_label_2.txt'
# input_file = '/home/zero/mnt/sda6/split_data_instance/MIAO-MAIZE/LD145-6-3-1.txt'
# input_path = input_file.split('/')[0:-2]
# input_file =  '/home/zero/mnt/sda6/PV_TEST/ptomato/split/select/032211-11_label_2.txt'
# output_path = './maize_2/seg-qu+pu'
# output_path = './Maize-miao/1.0/seg-qu+pu'
# output_path = '/home/zero/mnt/sda6/split_data_instance/MIAO-MAIZE/cluster'
# output_path = '/home/zero/mnt/sda6/split_data_instance/Cabbage/rm_4-3/down/split/leaf/cluster/分层的多尺度谱聚类/spec'

# output_path = '/home/zero/mnt/sda6/new_work/syau_single_maize/leaf/cluster'
# output_path ='/home/zero/mnt/sda6/PV_TEST/ptomato/split/select/tomato'
filename = input_file.split("/")[-1].split('.')[0]  #.split('_')[0]
# os.makedirs(output_path, exist_ok=True)
# output_file = os.path.join(output_path, filename)
# num_clusters = 3  # 设定簇的数量

# 读取点云数据
raw_points = read_point_cloud(input_file)
centroid,m,xyz = pc_normalize(raw_points[:, 0:3])
# xyz = raw_points[:, 0:3]

# output_init_file = output_file + "init_002-1.txt"
# renumber_path = os.path.join(output_path, "renumber")
# os.makedirs(renumber_path, exist_ok=True)
# renumber_region_points = os.path.join(renumber_path, filename) + "init_002-1.txt"
# renumber_spe_file = os.path.join(renumber_path, filename) + "-mid-1.txt"
# merged_points = os.path.join(renumber_path, filename) + "-merged-1.txt"
# renumber_merged_points = os.path.join(renumber_path, filename) + "-r-merged-1.txt"
# reversed_points = os.path.join(renumber_path, filename) + "-reversed.txt"


# 进行区域增长聚类
#Maize  0.011 #0.031#0.015 #0.016 #0.02
# tmp_init_path = "/home/zero/mnt/sda6/split_data_instance/ptomato/cluster_leaf/tmp_init"
# tmp_init_file = os.path.join(tmp_init_path, filename + ".txt")
# output_init_path = "/home/zero/mnt/sda6/split_data_instance/ptomato/cluster_leaf/init_result"
# output_init_file = os.path.join(output_init_path, filename + ".txt")

# distance_threshold= 0.02
# segments = region_growing(xyz, distance_threshold,50)
# save_segmented_results_with_labels(segments, tmp_init_file)
# region_points = read_point_cloud(tmp_init_file)
# region_points = renumber_labels(region_points)
# save_point_cloud_with_labels(output_init_file, region_points[:,0:3], region_points[:,-1])
# print(np.unique(region_points[:,-1]))




### 谱聚类处理
# mid_points = read_point_cloud(renumber_region_points)
# init_file = "/home/zero/mnt/sda6/split_data_instance/Soybean/pred/leaf/cluster/renumber/032212-77init_004.txt"
# mid_points = read_point_cloud(init_file)
# mid_save_path = "/home/zero/mnt/sda6/split_data_instance/Soybean/pred/leaf/cluster/renumber/spe2"
# # # # #####pt-11:0.002  pt-19:0.005  pt-41:0.0037
# # #
# # # ####pt-11:0.0018
# # ####cabbage 0.005
# mid_labels = spectral_clustering_later_n(mid_points[:, 0:3], mid_points[:, 3], n_neighbors=64, max_k=5,
#                                          variance_threshold=0.0012 )  #使用方差  25  #0.0009  #0.0025  #0.005
# mid_labels = spectral_clustering_later_lk(mid_points[:,0:3], mid_points[:,3],n_neighbors=24,max_k=7,s_threshold=0.60) #  0.45  0.25#使用轮廓系数
# unique_mid_label = np.unique(mid_labels)
# print(unique_mid_label)
# print(len(unique_mid_label))
# save_point_cloud_with_labels(renumber_spe_file, mid_points[:, 0:3], mid_labels)
# print(f'谱聚类结果已保存到 {renumber_spe_file}')


### 部分簇合并操作
# spec_points=read_point_cloud(renumber_spe_file)   #renumber_region_points
# spec_xyz,spec_labels = spec_points[:,0:3], spec_points[:,3]
# spec_clusters = separate_clusters(spec_xyz, spec_labels)  #基于簇质心的距离来进行簇的合并
# merged_clusters = merge_clusters(spec_clusters, distance_threshold=0.12)   #基于簇质心的距离来进行簇的合并
# save_merged_clusters(merged_clusters,merged_points)  #基于簇质心的距离来进行簇的合并
# merged_clusters, merged_labels = merge_clusters_based_on_neighbors(spec_xyz,spec_labels, n_neighbors=3, merge_threshold=34)  #基于簇共享邻居点的数量来进行簇的合并
# save_point_cloud_with_labels(merged_points, merged_clusters, merged_labels)  #基于共享邻居点来进行簇的合并
# merged_clusters, merged_labels =merge_clusters_based_on_mhd(renumber_region_points,0.4) #基于曼哈顿距离来进行簇的合并
# save_point_cloud_with_labels(merged_points, merged_clusters, merged_labels)  #基于曼哈顿来进行簇的合并



#标签对齐操作
# merged_data = read_point_cloud(merged_points)  #renumber_spe_file  merged_points  renumber_region_points
# renumber_merged_data = renumber_labels(merged_data)
# save_point_cloud_with_labels(renumber_merged_points,renumber_merged_data[:,0:3], renumber_merged_data[:,-1])
# print(np.unique(renumber_merged_data[:,-1]))
# #
# #
# #
# # ####反归一化,将叶片点云实例分割结果与茎干点云合并

renumber_merged_points = "/mnt/data1/split_data_instance/Cabbage/rm_4-3/down/split/leaf/cluster/renumber/spe/mvs_1116_07-mid-1.txt"
def reverse_normalize(pc, centroid, scale):
    """将点云从归一化状态恢复到原始状态"""
    pc = pc * scale
    pc = pc + centroid
    return pc

nor_points  = read_point_cloud(renumber_merged_points)  #renumber_region_points  #renumber_merged_points #renumber_merged_points  #renumber_spe_file   #reversed_points
nor_xyz,nor_label = nor_points[:, 0:3], nor_points[:, 3]
rever_xyz = reverse_normalize(nor_xyz, centroid,m)
rever_points = np.concatenate((rever_xyz,nor_label.reshape(-1,1)), axis=1)
# ###save_point_cloud_with_labels(reversed_points,rever_xyz, nor_label)
#
stem_path = '/mnt/data1/split_data_instance/Cabbage/rm_4-3/down/split/stem'
stem_file = os.path.join(stem_path, filename) + '.txt'  #+ '_label_1.txt'
stem_points = read_point_cloud(stem_file)
stem_xyz = stem_points[:, 0:3]
stem_label = stem_points[:, 3]   #pred  label
stem_point = np.concatenate((stem_xyz, stem_label.reshape(-1,1)), axis=1)
all_points = np.vstack((stem_point, rever_points))
all_points_path = '/mnt/data1/split_data_instance/Cabbage/rm_4-3/down/split/leaf/cluster/renumber/all'
os.makedirs(all_points_path, exist_ok=True)
all_points_file = os.path.join(all_points_path, filename) + '.txt'
save_point_cloud_with_labels(all_points_file,all_points[:,0:3], all_points[:,-1])
