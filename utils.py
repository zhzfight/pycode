import glob
import math
import os
import re
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

def computeTimeIntervalMatrix(time_seq,size):
    time_matrix = np.zeros([size, size], dtype=np.int32)
    shift=size-len(time_seq)
    for i in range(len(time_seq)):
        for j in range(len(time_seq)):
            span = abs(time_seq[i]-time_seq[j])
            time_matrix[i][j+shift]=span
    return time_matrix
def data_preprocessing(path,drop_poi,drop_user,split_time,seqlen_thre):
    df = pd.read_csv(path)
    # drop the different timezone
    df=df.drop(df[df['timezoneOffset']!=df['timezoneOffset'].value_counts().idxmax()].index)
    #
    df=df[df["venueId"].isin(df["venueId"].value_counts()[df["venueId"].value_counts() >= drop_poi].index)]
    df=df[df["userId"].isin(df["userId"].value_counts()[df["userId"].value_counts() >= drop_user].index)]
    poi_ids = list(set(df['venueId'].tolist()))
    poi_ids2idx=dict(zip(poi_ids,range(1,len(poi_ids)+1)))
    cat_ids = list(set(df['venueCategoryId'].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(1,len(cat_ids)+1)))
    user_ids = list(set(df['userId'].to_list()))
    user_id2idx_dict = dict(zip(user_ids, range(1,len(user_ids)+1)))
    df['venueId']=df['venueId'].map(poi_ids2idx)
    df['venueCategoryId']=df['venueCategoryId'].map(cat_id2idx_dict)
    df['userId']=df['userId'].map(user_id2idx_dict)
    df['traj_id']=''
    num_pois=len(set(df['venueId'].tolist()))
    num_cats=len(set(df['venueCategoryId'].tolist()))
    num_users=len(set(df['userId'].tolist()))
    def convert_to_datetime(utctimestamp):
        utcdate = datetime.datetime.strptime(utctimestamp, "%a %b %d %H:%M:%S %z %Y") # 转换成UTC日期和时间对象
        return utcdate
    df['datetime']=df['utcTimestamp'].apply(convert_to_datetime)
    def time_feature(utctimestamp):
        utcdate = datetime.datetime.strptime(utctimestamp, "%a %b %d %H:%M:%S %z %Y") # 转换成UTC日期和时间对象
        normal_time_inday=(utcdate.time().hour*3600+utcdate.time().minute*60+utcdate.time().second)/86400
        return utcdate.weekday()+normal_time_inday
    df['time_feature']=df['utcTimestamp'].apply(time_feature)
    df.drop(['venueCategory','latitude','longitude','timezoneOffset','utcTimestamp'],axis=1,inplace=True)
    df=df.sort_values(by='userId',kind='mergeSort')
    seq_no=0
    prev_user=-1
    prev_datetime=None
    for i in tqdm(range(len(df))):
        cur_user=df['userId'].iloc[i]
        if cur_user!=prev_user:
            seq_no=1
            prev_user=cur_user
            prev_datetime=df['datetime'].iloc[i]
            df['traj_id'].iloc[i]=str(cur_user)+'_'+str(seq_no)
            continue
        if (df['datetime'].iloc[i]-prev_datetime).total_seconds()>split_time:
            seq_no+=1
        df['traj_id'].iloc[i]=str(cur_user)+'_'+str(seq_no)
        prev_datetime=df['datetime'].iloc[i]
    # drop seqlen < 3
    df=df.groupby('traj_id').filter(lambda x: len(x) >= seqlen_thre)
    return df,num_pois,num_cats,num_users,df.groupby('traj_id').size().max(),df.groupby('traj_id').size().mean()
def split_df(df):
    df1=df.copy(deep=True)
    for name,ugroup in df.groupby(['userId']):
        df.drop(df1[(df1['traj_id']==ugroup.tail(1)['traj_id'].values[0]) & (df1['userId']==name)].index,inplace=True)
    for name,ugroup in df1.groupby(['userId']):
        df1.drop(df1[(df1['traj_id']!=ugroup.tail(1)['traj_id'].values[0]) & (df1['userId']==name)].index,inplace=True)
    return df,df1

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
