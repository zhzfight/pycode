import glob
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import random
from geographiclib.geodesic import Geodesic
from tqdm.contrib.concurrent import process_map

geod = Geodesic.WGS84
def compute_relative_time_matrix(t1, t2,remainder):
    t1 = np.array(t1)
    t2 = np.array(t2)
    matrix = (t1[:, None] - t2) % remainder + 1
    return np.tril(matrix)

def split_list(a_list, x):
    # 计算每份的长度和余数
    if x > len(a_list):
        x = len(a_list)
    k, m = divmod(len(a_list), x)
    # 使用列表生成器返回x份列表
    return [a_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(x)]

def to1(weights,adjOrDis):
    if adjOrDis=='adj':
        total = sum(weights)
        weights = [w / total for w in weights]
    else:
        for i in range(len(weights)):
            weights[i]=math.pow(weights[i]+10,1/2)
        weights=[1/i for i in weights]
        weights=[i/sum(weights) for i in weights]
    return weights

# 随机选择一个邻居节点
def choose_neighbor(graph, node,adjOrDis):
    if node not in graph:
        return None
    neighbors = [each[0] for each in graph[node]]
    weights = [each[1] for each in graph[node]]
    # 归一化权重
    weights=to1(weights,adjOrDis)
    # 根据权重分布随机选择邻居节点
    return random.choices(neighbors, weights)[0]

# 进行一次随机游走
def random_walk_with_restart(graph, start_node, restart_prob,num_walks,adjOrDis):
    adj_list = []
    current_node = start_node
    while True:
        if len(adj_list)>= num_walks:
            break
        p = random.random()
        if  p < restart_prob: # 以一定概率重启
            current_node=start_node
        else: # 否则继续游走
            nei = choose_neighbor(graph, current_node,adjOrDis)
            if nei is None:
                adj_list.append(start_node)
                current_node=start_node
                continue
            current_node=nei
            adj_list.append(current_node)

    return adj_list


def sample_neighbors(graph,nodes,restart_prob,num_walks,adjOrDis):
    res=[]
    for node in nodes:
        neighbors=random_walk_with_restart(graph,node,restart_prob,num_walks,adjOrDis)
        res.append(neighbors)
    return res

def get_node_geo_context_neighbors(index, nodes,geos, geo_k,geo_dis):
    neighbors = []
    for i in range(len(nodes)):
        if i==index:
            continue
        dis = geod.Inverse(geos[i][1], geos[i][0], geos[index][1], geos[index][0])['s12']
        neighbors.append((nodes[i],dis))
    sorted_data = sorted(neighbors, key=lambda x: x[1])
    for i in range(geo_k):
        if sorted_data[i][1]>geo_dis:
            return sorted_data[:i]
    return sorted_data[:geo_k]
# 定义一个函数，获取所有节点的邻居列表
def get_all_nodes_neighbors(nodes,geos,geo_k,geo_dis):
    ls=len(nodes)
    result = process_map(get_node_geo_context_neighbors, range(ls), [nodes] * ls,[geos]*ls,
                         [geo_k] * ls,[geo_dis]*ls, max_workers=None, chunksize=1)
    dis={}
    for i in range(ls):
        if len(result[i])==0:
            continue
        dis[nodes[i]]=result[i]
    return dis

def adj_list(raw_A,raw_X,geo_dis):
    raw_A=np.copy(raw_A).astype(np.float32)
    raw_X=np.copy(raw_X)
    # 假设邻接矩阵是一个二维数组matrix
    n = len(raw_A)  # 邻接矩阵的行数和列数
    adj_list = [[] for _ in range(n)]
    for i in tqdm(range(n)):
        for j in range(n):
            if raw_A[i][j] > 0:
                adj_list[i].append((j,raw_A[i][j]))

    nodes = [tuple(row) for row in np.asarray(raw_X[:, [3,2]])]
    dis =get_all_nodes_neighbors(nodes,geo_dis)

    return adj_list,dis

def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def get_normalized_features(X):
    # X.shape=(num_nodes, num_features)
    means = np.mean(X, axis=0)  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, -1))
    stds = np.std(X, axis=0)  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, -1))
    return X, means, stds


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


def array_round(x, k=4):
    # For a list of float values, keep k decimals of each element
    return list(np.around(np.array(x), k))
