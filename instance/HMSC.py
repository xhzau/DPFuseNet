import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
import os
from sklearn.metrics import silhouette_samples
# ------------------------------------------------------------
# 工具：簇内方差（trace(Cov)/3），与 variance_threshold 配合
# ------------------------------------------------------------
def compute_cluster_variance(subset):
    center = np.mean(subset, axis=0)
    distances = np.linalg.norm(subset - center, axis=1)
    variance = np.var(distances)
    return variance

# ------------------------------------------------------------
# 单尺度相似矩阵：互近邻 + 自适应带宽高斯核 + SNN
#   w_ij = exp( -||xi-xj||^2 / (σ_i σ_j) ) * ( |N(i)∩N(j)|/k )^alpha
#   不考虑各向异性（法向）；仅用几何距离 + SNN
# ------------------------------------------------------------
def _construct_similarity_single(points, n_neighbors=16, snn_alpha=1.5):
    """
    返回 csr_matrix（已对称化、去自环）
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return csr_matrix((0, 0))

    k = min(max(2, n_neighbors), n)
    knn = NearestNeighbors(n_neighbors=k).fit(pts)
    dists, idxs = knn.kneighbors(pts)
    neigh_sets = [set(row) for row in idxs]

    # 自适应带宽：每点 σ_i = 第 k 近邻距离（避免为 0）
    sig = dists[:, -1].copy()
    sig[sig < 1e-6] = 1e-6

    row_idx, col_idx, weights = [], [], []
    for i in range(n):
        for jpos, j in enumerate(idxs[i]):
            if i == j:
                continue
            # 互近邻：仅当 i∈N(j) 且 j∈N(i) 时连边
            if i in neigh_sets[j]:
                dij2 = float(np.sum((pts[i] - pts[j]) ** 2))
                w = np.exp(- dij2 / (sig[i] * sig[j]))  # 自适应带宽高斯核
                row_idx.append(i); col_idx.append(j); weights.append(w)

    # SNN 修正：共享近邻越多，权重越大
    if snn_alpha and snn_alpha > 0.0 and len(weights) > 0:
        for t in range(len(weights)):
            i, j = row_idx[t], col_idx[t]
            shared = len(neigh_sets[i].intersection(neigh_sets[j]))
            snn = (shared / float(k)) ** snn_alpha
            weights[t] *= snn

    A = csr_matrix((weights, (row_idx, col_idx)), shape=(n, n))
    A = 0.5 * (A + A.T)   # 对称化
    A.setdiag(0.0)        # 去自环
    A.eliminate_zeros()
    return A

# ------------------------------------------------------------
# 多尺度相似矩阵融合（逐边精确 max/sum/mean）
#   仅变化：k（近邻数）与 SNN α；均不使用各向异性
# ------------------------------------------------------------
def construct_similarity_matrix_multiscale(points,
                                           n_neighbors_list=(8, 16, 24),
                                           snn_alpha_list=(1.5, 1.5, 1.5),
                                           agg='max'):
    """
    多尺度融合：
      A_s = _construct_similarity_single(points, k_s, alpha_s)
    聚合：
      - 'max'  → 逐边取最大（保守，保留任一尺度的强边）
      - 'sum'  → 逐边求和（激进，叠加强边）
      - 'mean' → 逐边平均（平滑）
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return csr_matrix((0, 0))

    # 对齐长度
    L = min(len(n_neighbors_list), len(snn_alpha_list))
    n_neighbors_list = list(n_neighbors_list)[:L]
    snn_alpha_list   = list(snn_alpha_list)[:L]

    mats = []
    for k, alpha in zip(n_neighbors_list, snn_alpha_list):
        A = _construct_similarity_single(pts, n_neighbors=int(k), snn_alpha=float(alpha))
        mats.append(A)

    if not mats:
        return csr_matrix((n, n))

    # 逐边融合（精确）：用 dict 聚合
    if agg not in ('max', 'sum', 'mean'):
        agg = 'max'
    edge_map = {}  # key=(i,j, i<=j), val=list of weights
    for A in mats:
        coo = A.tocoo()
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if i == j:
                continue
            key = (int(i), int(j)) if i <= j else (int(j), int(i))
            edge_map.setdefault(key, []).append(float(w))

    rows, cols, data = [], [], []
    for (i, j), ws in edge_map.items():
        if agg == 'max':
            wij = float(np.max(ws))
        elif agg == 'sum':
            wij = float(np.sum(ws))
        else:  # 'mean'
            wij = float(np.mean(ws))
        rows.append(i); cols.append(j); data.append(wij)
        if i != j:
            rows.append(j); cols.append(i); data.append(wij)

    A_ms = csr_matrix((data, (rows, cols)), shape=(n, n))
    A_ms.setdiag(0.0)
    A_ms.eliminate_zeros()
    return A_ms

def compute_silhouette(points, labels):
    silhouette_vals = silhouette_samples(points, labels)
    return silhouette_vals

# ------------------------------------------------------------
# 主函数：多尺度相似矩阵 + 谱聚类（eigengap + KMeans）+ 重编号
#   与你的风格保持一致；仅更换相似矩阵构建为“多尺度 + SNN”
# ------------------------------------------------------------
def spectral_clustering_later_n(points, initial_labels,
                                max_k=6,
                                variance_threshold=0.01,
                                silhouette_threshold=0.6,
                                n_neighbors_list=(8, 16, 24),
                                snn_alpha_list=(1.5, 1.5, 1.5),
                                agg='max',
                                random_state=42):
    """
    对每个初始簇：
      1) 计算簇内方差，若 > variance_threshold 则细分
      2) 相似矩阵 = construct_similarity_matrix_multiscale(points_sub, n_neighbors_list, snn_alpha_list, agg)
      3) 归一化拉普拉斯 → 取前 m 特征向量 → eigengap 选 k
      4) KMeans(k) → 子标签回填
      5) 全局重编号（从 2 开始）
    说明：
      - 形参 n_neighbors/sigma 仅为兼容原接口，这里不使用 sigma，
        相似矩阵完全由多尺度构建（含自适应带宽 + SNN）。
    """
    points = np.asarray(points, dtype=float)
    labels0 = np.asarray(initial_labels, dtype=int)
    final_labels = np.zeros_like(labels0)

    silhouette_vals = compute_silhouette(points, initial_labels)

    for label in np.unique(labels0):
        subset = points[labels0 == label]
        variance = compute_cluster_variance(subset)
        subset_silhouette_vals = silhouette_vals[initial_labels == label]
        avg_silhouette = np.mean(subset_silhouette_vals)
        if variance > variance_threshold and len(subset) > 2:
        # if avg_silhouette < silhouette_threshold and len(subset) > 2:

            # —— 多尺度相似矩阵（含 SNN；不含各向异性）——
            A = construct_similarity_matrix_multiscale(
                subset,
                n_neighbors_list=n_neighbors_list,
                snn_alpha_list=snn_alpha_list,
                agg=agg
            )
            deg = np.array(A.sum(axis=1)).ravel()
            if len(deg) == 0:
                final_labels[labels0 == label] = label * max_k
                continue
            deg[deg < 1e-12] = 1e-12
            Dinv2 = diags(1.0 / np.sqrt(deg))
            Lsym = csr_matrix(np.eye(len(subset))) - Dinv2 @ A @ Dinv2

            m = max(2, min(max_k, len(subset) - 1))
            try:
                eigvals, eigvecs = eigsh(Lsym, k=m, which='SM')
            except Exception:
                final_labels[labels0 == label] = label * max_k
                continue
            gaps = np.diff(np.sort(eigvals))
            k = int(np.argmax(gaps) + 1) if len(gaps) >= 1 else 1
            if k >= 2:
                emb = normalize(eigvecs[:, :k])
                kmeans = KMeans(n_clusters=k, n_init=20, random_state=random_state)
                sub_labels = kmeans.fit_predict(emb)
                final_labels[labels0 == label] = sub_labels + label * max_k
            else:
                final_labels[labels0 == label] = label * max_k
        else:
            final_labels[labels0 == label] = label * max_k
    # 全局重编号（与原实现一致：从 2 开始）
    final_unique_labels = np.unique(final_labels)
    new_labels = {old_label: new_label for new_label, old_label in enumerate(final_unique_labels, start=2)}
    renumbered_labels = np.vectorize(new_labels.get)(final_labels)
    return renumbered_labels


def spectral_clustering_later_n_tiered(points,
                                       initial_labels,
                                       # 细分判据（任选其一或两个都开）
                                       use_variance=True,
                                       variance_threshold=0.01,
                                       use_silhouette=False,
                                       silhouette_threshold=0.6,
                                       # 分层的 max_k（大/中/小）
                                       max_k_large=8,
                                       max_k_mid=6,
                                       max_k_small=4,
                                       max_sizes=[2000,1000,50],
                                       # 多尺度相似度参数（与你现有一致）
                                       n_neighbors_list=(8, 16, 24),
                                       snn_alpha_list=(1.5, 1.5, 1.5),
                                       agg='max',
                                       random_state=42):
    """
    多层次谱聚类（分层 max_k）：
      1) 找到符合细分判据的簇集合 cands
      2) 按簇大小从大到小排序，并分成最多三层（不足三就按数量分层）
      3) 各层使用不同的 max_k 做谱聚类细分
      4) 用不冲突的全局偏移写回，最后统一重编号

    注意：
      - 判据默认用“方差 > variance_threshold”；你也可开启 silhouette 判据（平均簇内轮廓 < 阈值）
      - eigengap 自动选 k ∈ [1, per-cluster max_k]；若 k=1 则不细分
    """
    points = np.asarray(points, dtype=float)
    labels0 = np.asarray(initial_labels, dtype=int)
    final_labels = -np.ones_like(labels0, dtype=int)  # 先填 -1，避免冲突
    uniq = np.unique(labels0)
    sizes = {int(l): int(np.sum(labels0 == l)) for l in uniq}
    # 可选：预先计算全局轮廓分数（与你现有风格一致）
    sil_vals = None
    if use_silhouette:
        try:
            from sklearn.metrics import silhouette_samples
            sil_vals = silhouette_samples(points, labels0) if len(uniq) > 1 else None
        except Exception:
            sil_vals = None
    # 1) 先确定“需要细分”的候选簇
    cands = []
    for lab in uniq:
        mask = (labels0 == lab)
        subset = points[mask]
        need = False
        # 判据 A：方差（推荐）
        if use_variance:
            var = compute_cluster_variance(subset)
            if var > variance_threshold and len(subset) > 2:
                need = True
        # 判据 B：轮廓（可选）
        if use_silhouette and (sil_vals is not None):
            sub_sil = sil_vals[mask]
            avg_sil = float(np.mean(sub_sil)) if len(sub_sil) > 0 else 1.0
            if avg_sil < silhouette_threshold and len(subset) > 2:
                need = True
        if need:
            cands.append(int(lab))
    if len(cands) == 0:
        # 完全不细分，直接重编号后返回
        uniq_final = np.unique(labels0)
        remap = {old: new for new, old in enumerate(uniq_final, start=2)}
        return np.vectorize(remap.get)(labels0)

    # 2) 候选簇按大小降序排序，并分层（最多三层；不足就用实际层数）
    cands_sorted = sorted(cands, key=lambda l: sizes[int(l)], reverse=True)
    tiers = []
    tier_maxk = [max_k_large, max_k_mid, max_k_small]

    # 根据点数大小进行分层
    start_idx = 0
    for size_threshold, max_k in zip(max_sizes, tier_maxk):
        end_idx = len(cands_sorted)
        for idx in range(start_idx, len(cands_sorted)):
            if sizes[cands_sorted[idx]] < size_threshold:
                end_idx = idx
                break
        tiers.append((cands_sorted[start_idx:end_idx], max_k))
        start_idx = end_idx
    ####均匀会划分
    # T = min(3, len(cands_sorted))
    # # 计算层切分索引（均匀切三段；不足三则按数量切）
    # splits = []
    # if T == 1:
    #     splits = [len(cands_sorted)]
    #     tier_maxk_list = [max_k_large]
    # elif T == 2:
    #     splits = [int(np.ceil(len(cands_sorted) / 2.0)), len(cands_sorted)]
    #     tier_maxk_list = [max_k_large, max_k_mid]
    # else:
    #     a = int(np.ceil(len(cands_sorted) / 3.0))
    #     b = int(np.ceil(2 * len(cands_sorted) / 3.0))
    #     splits = [a, b, len(cands_sorted)]
    #     tier_maxk_list = [max_k_large, max_k_mid, max_k_small]
    # # 划出各层
    # tiers = []
    # start = 0
    # for s in splits:
    #     tiers.append(cands_sorted[start:s])
    #     start = s

    # 3) 非候选簇：直接抄回（占位，待最终统一重编号）
    for lab in uniq:
        if lab not in cands:
            final_labels[labels0 == lab] = lab  # 暂存原标签值

    # 4) 逐层细分
    next_base = int(np.max(labels0)) + 1  # 全局偏移，确保唯一
    for labs_in_tier, max_k in tiers:
        for lab in labs_in_tier:
            mask = (labels0 == lab)
            subset = points[mask]
    # next_base = int(np.max(labels0)) + 1  # 全局偏移，确保唯一
    # for tier_idx, labs_in_tier in enumerate(tiers):
    #     per_cluster_max_k = int(tier_maxk_list[tier_idx])
    #     for lab in labs_in_tier:
    #         mask = (labels0 == lab)
    #         subset = points[mask]
            if len(subset) < 3:
                final_labels[mask] = lab  # 点太少不细分
                continue
            # —— 多尺度相似矩阵（与你现有构图一致）——
            A = construct_similarity_matrix_multiscale(
                subset,
                n_neighbors_list=n_neighbors_list,
                snn_alpha_list=snn_alpha_list,
                agg=agg
            )
            deg = np.array(A.sum(axis=1)).ravel()
            if len(deg) == 0:
                final_labels[mask] = lab
                continue
            deg[deg < 1e-12] = 1e-12
            Dinv2 = diags(1.0 / np.sqrt(deg))
            # Lsym = I - D^{-1/2} A D^{-1/2}
            Lsym = csr_matrix(np.eye(len(subset))) - Dinv2 @ A @ Dinv2
            # per-cluster 的最大特征数 m，至少 2，且 < n
            m = max(2, min(max_k, len(subset) - 1))
            try:
                eigvals, eigvecs = eigsh(Lsym, k=m, which='SM')
            except Exception:
                final_labels[mask] = lab
                continue
            gaps = np.diff(np.sort(eigvals))
            k = int(np.argmax(gaps) + 1) if len(gaps) >= 1 else 1
            # 保护：k 不超过该层的 max_k，不超过样本-1
            k = int(max(1, min(k, max_k, len(subset) - 1)))

            if k >= 2:
                emb = normalize(eigvecs[:, :k])
                km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
                sub_labels = km.fit_predict(emb)
                # 全局唯一写回
                final_labels[mask] = sub_labels + next_base
                next_base += (int(np.max(sub_labels)) + 1)
            else:
                final_labels[mask] = lab

    # 5) 统一重编号（从 2 开始）
    uniq_final = np.unique(final_labels)
    remap = {old: new for new, old in enumerate(uniq_final, start=2)}
    refined = np.vectorize(remap.get)(final_labels)
    return refined


def save_point_cloud_with_labels(file_path, points, labels):
    data_with_labels = np.hstack((points, labels.reshape(-1, 1)))
    np.savetxt(file_path, data_with_labels, fmt='%.6f')


if __name__ == "__main__":
    #input_file = '/home/zero/mnt/sda6/split_data_instance/ptomato/cluster_leaf/init_result/032211-44.txt'
    # input_file = '/mnt/data1/split_data_instance/Corn/corn01/select/init/renumber/ini/032213-137_0.02.txt'
    # input_file = '/mnt/data1/new_work/syau_single_maize/leaf/cluster/renumber/LD145-10-6-1init_004.txt'
    input_file ="/mnt/data1/split_data_instance/potato/leaf/init/renumber/032215-82_0.02.txt"

    output_path = '/mnt/data1/split_data_instance/potato/leaf/init/renumber/spe'
    filename = input_file.split("/")[-1].split('.')[0]  # .split('_')[0]
    os.makedirs(output_path, exist_ok=True)
    # output_file = os.path.join(output_path, filename)
    renumber_spe_file = os.path.join(output_path, filename) + ".txt"
    mid_points = np.loadtxt(input_file)
    xyz = mid_points[:, 0:3]
    initial_labels = mid_points[:, -1]


    # refined = spectral_clustering_later_n(
    #     xyz,
    #     initial_labels=initial_labels,
    #     max_k=7,
    #     variance_threshold=0.001,
    #     silhouette_threshold = 0.5,
    #     n_neighbors_list=(8, 16, 24),
    #     snn_alpha_list=(1.8, 1.8, 1.5),
    #     agg='max',
    #     random_state=42
    # )
    refined = spectral_clustering_later_n_tiered(xyz,
                                       initial_labels,
                                       use_variance=True,
                                       variance_threshold= 0.0021,
                                       use_silhouette=False,
                                       silhouette_threshold=0.3,
                                       # 分层的 max_k（大/中/小）
                                       max_k_large=3,  #8  10
                                       max_k_mid=3,
                                       max_k_small=3,
                                       max_sizes=[3000, 1500, 50],
                                       # 多尺度相似度参数（与你现有一致）
                                       n_neighbors_list=(16, 32, 84),  #8, 16, 24  #16, 32, 64
                                       snn_alpha_list=(1.5, 1.5, 1.5),
                                       agg='max',
                                       random_state=42)


    print("簇数：", len(np.unique(refined)))
    save_point_cloud_with_labels(renumber_spe_file, xyz, refined)
    print("谱聚类结果保存于：",renumber_spe_file)
