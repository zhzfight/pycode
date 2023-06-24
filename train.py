import collections
import logging
import logging
import math
import os
import pathlib
import pickle
import zipfile
from pathlib import Path
import multiprocessing
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import UserEmbeddings, CategoryEmbeddings, TimeIntervalAwareTransformer, PoiEmbeddings, TimeEmbeddings, \
    GraphSAGE, TransformerModel
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, adj_list, split_list, random_walk_with_restart, \
    get_all_nodes_neighbors, compute_relative_time_matrix
import threading


def train(args):
    print(
        "three mode you can choose,1.poi-sage 2.poi 3.sage\n"
        "you can set mode through args.embed_mode.\n"
        "three dataset you can choose in this code, the first is NYC used in GETNext,"
        "But I think this is an overprocessed dataset. the second and third is NYC launched by Yang DingQi."
        "this two dataset are very big, and For these two datasets, I didn’t do too much extra preprocessing."
        "If you use the Sage mode, because GraphSAGE requires sampling, this will cause the program to run very slowly. "
        "But this is because I used Manager().Queue in the producer-consumer model for inter-process communication, "
        "and its performance is far inferior to that of multiprocessing.Queue. Because my implementation is not very "
        "elegant, it causes multiprocessing.Queue to require many threads, which results in not enough threads when "
        "there are many POIs. If the number of POIs is relatively small, the solution using multiprocessing.Queue has "
        "a flying speed. If you need this, you can contact me."
        'have a nice day.')
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    df = pd.read_csv(args.dataset)
    df['POI_id'] = pd.factorize(df['POI_id'])[0]
    df['POI_catid'] = pd.factorize(df['POI_catid'])[0]
    df['user_id'] = pd.factorize(df['user_id'])[0]
    df['datetime'] = pd.to_datetime(df['local_time'])
    df['hour_of_week'] = df['datetime'].dt.dayofweek * 24 + df['datetime'].dt.hour
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    poi_num = len(set(df['POI_id'].to_list()))
    cat_num = len(set(df['POI_catid'].to_list()))
    user_num = len(set(df['user_id'].to_list()))
    print(f"poi_num: {poi_num}, cat_num: {cat_num}, user_num: {user_num}")
    poi2cat = df.set_index('POI_id')['POI_catid'].to_dict()

    # %% ====================== Define Dataset ======================

    class produceSampleProcess(multiprocessing.Process):
        def __init__(self, tasks, node_dicts, adj_list, restart_prob, num_walks, threshold, adjOrdis,  id):
            super().__init__()
            self.tasks = tasks
            self.node_dicts = node_dicts
            self.threshold = threshold
            self.adjOrdis = adjOrdis
            self.id = id
            self.adj_list = adj_list
            self.restart_prob = restart_prob
            self.num_walks = num_walks
            self.count_dict = {key: threshold for key in tasks}

        def run(self):
            while True:
                for node in self.tasks:
                    if self.node_dicts[node].empty():
                        for _ in range(self.count_dict[node]):
                            random_walk = random_walk_with_restart(self.adj_list, node, self.restart_prob,
                                                                   self.num_walks,
                                                                   self.adjOrdis)
                            self.node_dicts[node].put(random_walk)
                        if self.count_dict[node]+threshold>2000:
                            self.count_dict[node]=2000
                        else:
                            self.count_dict[node]+=threshold


    pois_in_train = set()

    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, df, time_threshold):
            self.df = df.copy()
            self.users = []
            self.input_seqs = []
            self.label_seqs = []
            self.input_seq_w_timeMatrixes = []
            self.input_seq_h_timeMatrixes = []
            self.label_seq_w_timeMatrixes = []
            self.label_seq_h_timeMatrixes = []
            input_end_idx = -3
            self.time_threshold = time_threshold
            label_end_idx = input_end_idx + 1
            for user in tqdm(set(df['user_id'].tolist())):
                user_df = df[df['user_id'] == user]
                user_df = user_df.sort_values(by='datetime')
                poi_ids = user_df['POI_id'].to_list()
                time_bins = user_df['hour_of_week'].to_list()
                h = user_df['hour_of_day'].to_list()
                w = user_df['day_of_week'].to_list()
                dt = user_df['datetime'].to_list()
                if len(poi_ids) < 5:
                    continue

                input_start_idx = max(len(poi_ids) - 3 - args.max_seq_len, 0)
                label_start_idx = input_start_idx + 1
                self.users.append(user)
                self.input_seqs.append(list(
                    zip(poi_ids[input_start_idx:input_end_idx], time_bins[input_start_idx:input_end_idx],
                        h[input_start_idx:input_end_idx], w[input_start_idx:input_end_idx],
                        dt[input_start_idx:input_end_idx], )))
                input_h_matrix = compute_relative_time_matrix(h[input_start_idx:input_end_idx],h[input_start_idx:input_end_idx], 24)
                input_w_matrix = compute_relative_time_matrix(w[input_start_idx:input_end_idx],w[input_start_idx:input_end_idx], 7)
                self.input_seq_h_timeMatrixes.append(torch.LongTensor(input_h_matrix))
                self.input_seq_w_timeMatrixes.append(torch.LongTensor(input_w_matrix))
                self.label_seqs.append(list(
                    zip(poi_ids[label_start_idx:label_end_idx], time_bins[label_start_idx:label_end_idx],
                        h[label_start_idx:label_end_idx], w[label_start_idx:label_end_idx],
                        dt[label_start_idx:label_end_idx], )))
                label_h_matrix = compute_relative_time_matrix(h[label_start_idx:label_end_idx],h[input_start_idx:input_end_idx], 24)
                label_w_matrix = compute_relative_time_matrix(w[label_start_idx:label_end_idx],w[input_start_idx:input_end_idx], 7)
                self.label_seq_h_timeMatrixes.append(torch.LongTensor(label_h_matrix))
                self.label_seq_w_timeMatrixes.append(torch.LongTensor(label_w_matrix))

                pois_in_train.update(poi_ids[input_start_idx:input_end_idx])

        def get_adj(self):
            adj = collections.defaultdict(dict)
            for seq in self.input_seqs:
                pois = [each[0] for each in seq]
                dt = [each[4] for each in seq]
                for i in range(len(pois) - 1):
                    if dt[i + 1] - dt[i] < self.time_threshold:
                        if pois[i + 1] not in adj[pois[i]]:
                            adj[pois[i]][pois[i + 1]] = 1
                        else:
                            adj[pois[i]][pois[i + 1]] += 1
            res = {}
            for key, value in adj.items():
                res[key] = list(value.items())
            return res

        def get_X(self):
            def remap_checkIn_count(checkIn_count):
                if checkIn_count < 10:
                    return 0
                elif checkIn_count < 50:
                    return 1
                elif checkIn_count < 200:
                    return 2
                else:
                    return 3

            # 按照timestamp列排序
            df = self.df.sort_values(['user_id', 'datetime'])
            # 获取每个组中时间戳最大的两条记录的索引
            idx = df.groupby('user_id').tail(2).index
            # 删除指定索引的行
            df = df.drop(idx)

            pois = list(set(df['POI_id'].to_list()))
            geos = []
            X = np.zeros((poi_num, (4 + cat_num)), dtype=np.float32)
            print('node feats making')
            for poi in tqdm(pois):
                checkin_count = len(df[df['POI_id'] == poi])
                cat = df.loc[df['POI_id'] == poi, 'POI_catid'].iloc[0]
                longitude = df.loc[df['POI_id'] == poi, 'longitude'].iloc[0]
                latitude = df.loc[df['POI_id'] == poi, 'latitude'].iloc[0]
                X[poi][remap_checkIn_count(checkin_count)] = 1
                X[poi][cat + 4] = 1
                geos.append((longitude, latitude))
            return X, pois, geos

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (
                self.users[index], self.input_seqs[index], self.label_seqs[index], self.input_seq_h_timeMatrixes[index],
                self.input_seq_w_timeMatrixes[index], self.label_seq_h_timeMatrixes[index],
                self.label_seq_w_timeMatrixes[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.users = []
            self.input_seqs = []
            self.label_seqs = []
            self.input_seq_w_timeMatrixes = []
            self.input_seq_h_timeMatrixes = []
            self.label_seq_w_timeMatrixes = []
            self.label_seq_h_timeMatrixes = []
            input_end_idx = -2
            label_end_idx = input_end_idx + 1
            for user in tqdm(set(df['user_id'].tolist())):
                user_df = df[df['user_id'] == user]
                user_df = user_df.sort_values(by='datetime')

                poi_ids = user_df['POI_id'].to_list()
                time_bins = user_df['hour_of_week'].to_list()
                h = user_df['hour_of_day'].to_list()
                w = user_df['day_of_week'].to_list()
                if len(poi_ids) < 5:
                    continue
                if poi_ids[-2] not in pois_in_train:
                    continue
                input_start_idx = max(len(poi_ids) - 2 - args.max_seq_len, 0)
                label_start_idx = input_start_idx + 1
                self.users.append(user)
                self.input_seqs.append(list(
                    zip(poi_ids[input_start_idx:input_end_idx], time_bins[input_start_idx:input_end_idx],
                        h[input_start_idx:input_end_idx], w[input_start_idx:input_end_idx])))
                input_h_matrix = compute_relative_time_matrix(h[input_start_idx:input_end_idx],
                                                              h[input_start_idx:input_end_idx], 24)
                input_w_matrix = compute_relative_time_matrix(w[input_start_idx:input_end_idx],
                                                              w[input_start_idx:input_end_idx], 7)
                self.input_seq_h_timeMatrixes.append(torch.LongTensor(input_h_matrix))
                self.input_seq_w_timeMatrixes.append(torch.LongTensor(input_w_matrix))
                self.label_seqs.append(list(
                    zip(poi_ids[label_start_idx:label_end_idx], time_bins[label_start_idx:label_end_idx],
                        h[label_start_idx:label_end_idx], w[label_start_idx:label_end_idx])))
                label_h_matrix = compute_relative_time_matrix(h[label_start_idx:label_end_idx],
                                                              h[input_start_idx:input_end_idx], 24)
                label_w_matrix = compute_relative_time_matrix(w[label_start_idx:label_end_idx],
                                                              w[input_start_idx:input_end_idx], 7)
                self.label_seq_h_timeMatrixes.append(torch.LongTensor(label_h_matrix))
                self.label_seq_w_timeMatrixes.append(torch.LongTensor(label_w_matrix))

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (
                self.users[index], self.input_seqs[index], self.label_seqs[index], self.input_seq_h_timeMatrixes[index],
                self.input_seq_w_timeMatrixes[index], self.label_seq_h_timeMatrixes[index],
                self.label_seq_w_timeMatrixes[index])

    class TrajectoryDatasetTest(Dataset):
        def __init__(self, df):
            self.users = []
            self.input_seqs = []
            self.label_seqs = []
            self.input_seq_w_timeMatrixes = []
            self.input_seq_h_timeMatrixes = []
            self.label_seq_w_timeMatrixes = []
            self.label_seq_h_timeMatrixes = []
            input_end_idx = -1
            for user in tqdm(set(df['user_id'].tolist())):
                user_df = df[df['user_id'] == user]
                user_df = user_df.sort_values(by='datetime')
                poi_ids = user_df['POI_id'].to_list()
                time_bins = user_df['hour_of_week'].to_list()
                h = user_df['hour_of_day'].to_list()
                w = user_df['day_of_week'].to_list()
                if len(poi_ids) < 5:
                    continue
                if poi_ids[-1] not in pois_in_train:
                    continue
                input_start_idx = max(len(poi_ids) - 1 - args.max_seq_len, 0)
                label_start_idx = input_start_idx + 1
                self.users.append(user)
                self.input_seqs.append(list(
                    zip(poi_ids[input_start_idx:input_end_idx], time_bins[input_start_idx:input_end_idx],
                        h[input_start_idx:input_end_idx], w[input_start_idx:input_end_idx])))
                input_h_matrix = compute_relative_time_matrix(h[input_start_idx:input_end_idx],
                                                              h[input_start_idx:input_end_idx], 24)
                input_w_matrix = compute_relative_time_matrix(w[input_start_idx:input_end_idx],
                                                              w[input_start_idx:input_end_idx], 7)
                self.input_seq_h_timeMatrixes.append(torch.LongTensor(input_h_matrix))
                self.input_seq_w_timeMatrixes.append(torch.LongTensor(input_w_matrix))
                self.label_seqs.append(list(
                    zip(poi_ids[label_start_idx:], time_bins[label_start_idx:],
                        h[label_start_idx:], w[label_start_idx:])))
                label_h_matrix = compute_relative_time_matrix(h[label_start_idx:],
                                                              h[input_start_idx:input_end_idx], 24)
                label_w_matrix = compute_relative_time_matrix(w[label_start_idx:],
                                                              w[input_start_idx:input_end_idx], 7)
                self.label_seq_h_timeMatrixes.append(torch.LongTensor(label_h_matrix))
                self.label_seq_w_timeMatrixes.append(torch.LongTensor(label_w_matrix))

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (
                self.users[index], self.input_seqs[index], self.label_seqs[index], self.input_seq_h_timeMatrixes[index],
                self.input_seq_w_timeMatrixes[index], self.label_seq_h_timeMatrixes[index],
                self.label_seq_w_timeMatrixes[index])

    def time_collate_fn(batch):
        users, input_seqs, label_seqs, input_seq_h_matrices, input_seq_w_matrices, \
            label_seq_h_matrices, label_seq_w_matrices = zip(*batch)
        max_size = max([len(input_seq) for input_seq in input_seqs])
        input_seq_h_padded_matrices = [F.pad(matrix, (0, max_size - matrix.shape[0], 0, max_size - matrix.shape[1])) for
                                     matrix in input_seq_h_matrices]
        input_seq_w_padded_matrices = [F.pad(matrix, (0, max_size - matrix.shape[0], 0, max_size - matrix.shape[1])) for
                                     matrix in input_seq_w_matrices]
        label_seq_h_padded_matrices = [F.pad(matrix, (0, max_size - matrix.shape[0], 0, max_size - matrix.shape[1])) for
                                     matrix in label_seq_h_matrices]
        label_seq_w_padded_matrices = [F.pad(matrix, (0, max_size - matrix.shape[0], 0, max_size - matrix.shape[1])) for
                                     matrix in label_seq_w_matrices]
        return list(zip(users, input_seqs, label_seqs, input_seq_h_padded_matrices, \
            input_seq_w_padded_matrices, label_seq_h_padded_matrices, label_seq_w_padded_matrices))
    collate_fn=lambda x:x
    if not args.pure_transformer:
        collate_fn=time_collate_fn
    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(df, pd.Timedelta(6, unit='h'))
    val_dataset = TrajectoryDatasetVal(df)
    test_dataset = TrajectoryDatasetTest(df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=collate_fn)
    test_loader = DataLoader(val_dataset,
                             batch_size=args.batch,
                             shuffle=False, drop_last=False,
                             pin_memory=True, num_workers=args.workers,
                             collate_fn=collate_fn)
    adj = None
    dis = None
    X = None
    if args.embed_mode != 'poi':
        basename = os.path.basename(args.dataset)
        prefix, _ = os.path.splitext(basename)
        if os.path.exists(os.path.join(os.path.dirname(args.dataset), prefix+'_adj.pkl')):
            with open(os.path.join(os.path.dirname(args.dataset), prefix+'_adj.pkl'), 'rb') as f:  # 打开pickle文件
                adj = pickle.load(f)  # 读取字典
        else:
            adj = train_dataset.get_adj()
            with open(os.path.join(os.path.dirname(args.dataset), prefix+'_adj.pkl'), 'wb') as f:
                pickle.dump(adj, f)  # 把字典写入pickle文件

        if os.path.exists(os.path.join(os.path.dirname(args.dataset), prefix+'_dis.pkl')):
            with open(os.path.join(os.path.dirname(args.dataset), prefix+'_dis.pkl'), 'rb') as f:  # 打开pickle文件
                dis = pickle.load(f)  # 读取字典
            X = np.load(os.path.join(os.path.dirname(args.dataset), prefix+'_X.npy'))
        else:
            X, pois, geos = train_dataset.get_X()
            print('space neighbor table making, if you have multi cpus, it will be faster.')
            dis = get_all_nodes_neighbors(pois, geos, args.geo_k, args.geo_dis)
            with open(os.path.join(os.path.dirname(args.dataset), prefix+'_dis.pkl'), 'wb') as f:
                pickle.dump(dis, f)  # 把字典写入pickle文件
            np.save(os.path.join(os.path.dirname(args.dataset), prefix+'_X.npy'), X)
        average_adj_len = 0
        for k, v in adj.items():
            average_adj_len += len(v)
        average_adj_len = average_adj_len / len(adj)
        average_dis_len = 0
        for k, v in dis.items():
            average_dis_len += len(v)
        average_dis_len = average_dis_len / len(dis)
        print(f'adj {len(adj)} {average_adj_len} dis {len(dis)} {average_dis_len}')
    adj_dicts = None
    dis_dicts = None
    managers=None

    threshold = 10  # 队列大小阈值
    process_list=[]
    if args.embed_mode != 'poi':
        managers=[]
        managers_num=10
        for _ in range(managers_num):
            managers.append(multiprocessing.Manager())
        adj_dicts=[managers[i%managers_num].Queue() for i in range(poi_num)]
        dis_dicts=[managers[i%managers_num].Queue() for i in range(poi_num)]
        tasks = split_list([i for i in range(poi_num)], int(args.cpus / 2))
        for idx, task in enumerate(tasks):
            ap = produceSampleProcess(tasks=task, node_dicts=adj_dicts, adj_list=adj, restart_prob=args.restart_prob,
                                      num_walks=args.num_walks,
                                      threshold=threshold, adjOrdis='adj',  id=idx)
            ap.start()
            process_list.append(ap)
            dp = produceSampleProcess(tasks=task, node_dicts=dis_dicts, adj_list=dis, restart_prob=args.restart_prob,
                                      num_walks=args.num_walks,
                                      threshold=threshold, adjOrdis='dis',  id=idx)
            dp.start()
            process_list.append(dp)

    # %% ====================== Build Models ======================
    poi_id_embed_model = PoiEmbeddings(poi_num, args.poi_id_dim)
    poi_sage_embed_model = None
    if args.embed_mode != 'poi':
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device=args.device, dtype=torch.float)
        poi_sage_embed_model = GraphSAGE(input_dim=X.shape[1], embed_dim=args.poi_sage_dim,
                                         device=args.device, restart_prob=args.restart_prob, num_walks=args.num_walks,
                                         dropout=args.dropout, adj_dicts=adj_dicts, dis_dicts=dis_dicts)

    user_embed_model = UserEmbeddings(user_num, args.user_embed_dim)

    # %% Model3: Time Model
    time_embed_model = TimeEmbeddings(args.time_embed_dim)

    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(cat_num, args.cat_embed_dim)

    # %% Model6: Sequence model
    poi_embed_dim = 0
    if args.embed_mode == 'poi-sage':
        poi_embed_dim = args.poi_sage_dim + args.poi_id_dim
    elif args.embed_mode == 'sage':
        poi_embed_dim = args.poi_sage_dim
    else:
        poi_embed_dim = args.poi_id_dim
    args.seq_input_embed = poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    if args.pure_transformer:
        seq_model = TransformerModel(poi_num,
                                     cat_num,
                                     args.seq_input_embed,
                                     nhead=1,
                                     nhid=args.seq_input_embed,
                                     nlayers=2,
                                     device=args.device,
                                     dropout=args.dropout)
    else:
        seq_model = TimeIntervalAwareTransformer(num_poi=poi_num,
                                                 num_cat=cat_num,
                                                 nhid=args.seq_input_embed,
                                                 batch_size=args.batch,
                                                 device=args.device,
                                                 dropout=args.dropout, user_dim=args.user_embed_dim)

    # Define overall loss and optimizer
    if args.embed_mode == 'poi-sage':
        parameter_list = list(poi_id_embed_model.parameters()) + list(poi_sage_embed_model.parameters()) + \
                         list(user_embed_model.parameters()) + list(time_embed_model.parameters()) + list(
            cat_embed_model.parameters()) + list(seq_model.parameters())
    elif args.embed_mode == 'poi':
        parameter_list = list(poi_id_embed_model.parameters()) + \
                         list(user_embed_model.parameters()) + list(time_embed_model.parameters()) + list(
            cat_embed_model.parameters()) + list(seq_model.parameters())
    else:
        parameter_list = list(poi_sage_embed_model.parameters()) + \
                         list(user_embed_model.parameters()) + list(time_embed_model.parameters()) + list(
            cat_embed_model.parameters()) + list(seq_model.parameters())
    optimizer = optim.Adam(params=parameter_list,
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, mode, train_or_eval, freeze=False, poi_sage_embeddings=None,
                                 embedding_index=None):
        user = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_cat = [poi2cat[each] for each in input_seq]
        input_seq_time = [each[1] for each in sample[1]]
        if mode != 'poi':
            if mode == 'poi-sage':
                if train_or_eval == 'eval' or freeze:
                    poi_idxs = input_seq
                else:
                    poi_idxs = [embedding_index + idx for idx in range(len(input_seq))]
                poi_sage_embed = poi_sage_embeddings[poi_idxs]
                poi_id_embeded = poi_id_embed_model(torch.LongTensor(input_seq).to(args.device))
                poi_embed = torch.cat((poi_sage_embed, poi_id_embeded), dim=-1)
            else:
                if train_or_eval == 'eval' or freeze:
                    poi_idxs = input_seq
                else:
                    poi_idxs = [embedding_index + idx for idx in range(len(input_seq))]
                poi_embed = poi_sage_embeddings[poi_idxs]
        else:
            poi_embed = poi_id_embed_model(torch.LongTensor(input_seq).to(args.device))
        catid_embed = cat_embed_model(torch.LongTensor(input_seq_cat).to(args.device))
        timebin_embed = time_embed_model(torch.LongTensor(input_seq_time).to(args.device))
        user_embed = user_embed_model(torch.LongTensor([user]).to(args.device))
        user_embed = user_embed.repeat(poi_embed.shape[0], 1)
        embedded = torch.cat((poi_embed, catid_embed, timebin_embed, user_embed), dim=-1)

        return embedded

    # %% ====================== Train ======================
    if args.embed_mode == 'poi-sage':
        poi_sage_embed_model = poi_sage_embed_model.to(device=args.device)
        poi_id_embed_model = poi_id_embed_model.to(device=args.device)
    elif args.embed_mode == 'sage':
        poi_sage_embed_model = poi_sage_embed_model.to(device=args.device)
    elif args.embed_mode == 'poi':
        poi_id_embed_model = poi_id_embed_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf
    freeze_sage_embeddings = None
    freeze_set = False
    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        if args.embed_mode == 'poi-sage':
            poi_sage_embed_model.train()
            poi_id_embed_model.train()
        elif args.embed_mode == 'sage':
            poi_sage_embed_model.train()
        elif args.embed_mode == 'poi':
            poi_id_embed_model.train()
        freeze = args.freeze_sage and freeze_set

        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        seq_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []

        poi_sage_embeddings = None
        embedding_index = 0
        if freeze and args.mode != 'poi':
            for param in poi_sage_embed_model.parameters():
                param.requires_grad = False
            pois = [n for n in range(poi_num)]
            poi_sage_embed_model.setup(X, adj, dis)
            freeze_sage_embeddings = poi_sage_embed_model(torch.tensor(pois).to(args.device))
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            # For padding

            batch_seq_lens = []
            batch_seq_embeds = []
            batch_label_seqs = []
            batch_user = []
            batch_input_h_matrices=[]
            batch_input_w_matrices = []
            batch_label_h_matrices = []
            batch_label_w_matrices = []

            if freeze and args.mode != 'poi':
                poi_sage_embeddings = freeze_sage_embeddings
            elif args.embed_mode != 'poi':
                pois = [each[0] for sample in batch for each in sample[1]]
                poi_sage_embed_model.setup(X, adj, dis)
                poi_sage_embeddings = poi_sage_embed_model(torch.tensor(pois).to(args.device))
            # Convert input seq to embeddings
            for sample in batch:
                batch_user.append(sample[0])
                batch_input_h_matrices.append(sample[3])
                batch_input_w_matrices.append(sample[4])
                batch_label_h_matrices.append(sample[5])
                batch_label_w_matrices.append(sample[6])
                label_seq = [each[0] for each in sample[2]]
                batch_label_seqs.append(torch.LongTensor(label_seq).to(args.device))

                input_seq_embed = input_traj_to_embeddings(sample, args.embed_mode, 'train', freeze=freeze,
                                                           poi_sage_embeddings=poi_sage_embeddings,
                                                           embedding_index=embedding_index)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(label_seq))
                embedding_index += len(label_seq)
            embedding_index = 0
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_label_seqs, batch_first=True, padding_value=-1)
            batch_user_embedding = user_embed_model(torch.LongTensor(batch_user).to(args.device))


            # Feedforward
            if not args.pure_transformer:
                batch_input_h_matrices=torch.stack(batch_input_h_matrices).to(args.device)
                batch_input_w_matrices = torch.stack(batch_input_w_matrices).to(args.device)
                batch_label_h_matrices = torch.stack(batch_label_h_matrices).to(args.device)
                batch_label_w_matrices = torch.stack(batch_label_w_matrices).to(args.device)

            x = batch_padded.to(device=args.device)
            y_poi = label_padded_poi.to(device=args.device)
            y_pred_poi = seq_model(x, batch_seq_lens, batch_input_h_matrices, batch_input_w_matrices, batch_label_h_matrices,
                                   batch_label_w_matrices, batch_user_embedding)

            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)

            # Final loss
            loss = loss_poi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (10)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)

        # train end --------------------------------------------------------------------------------------------------------
        if args.embed_mode == 'poi-sage':
            poi_sage_embed_model.eval()
            poi_id_embed_model.eval()
        elif args.embed_mode == 'sage':
            poi_sage_embed_model.eval()
        elif args.embed_mode == 'poi':
            poi_id_embed_model.eval()

        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        poi_sage_embeddings = None
        if freeze and args.mode != 'poi':
            poi_sage_embeddings = freeze_sage_embeddings
        elif args.embed_mode != 'poi':
            pois = [n for n in range(poi_num)]
            poi_sage_embed_model.setup(X, adj, dis)
            poi_sage_embeddings = poi_sage_embed_model(torch.tensor(pois).to(args.device))
        for vb_idx, batch in enumerate(val_loader):

            # For padding
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_label_seqs = []
            batch_user = []
            batch_input_h_matrices = []
            batch_input_w_matrices = []
            batch_label_h_matrices = []
            batch_label_w_matrices = []

            # Convert input seq to embeddings
            for sample in batch:
                batch_user.append(sample[0])
                batch_input_h_matrices.append(sample[3])
                batch_input_w_matrices.append(sample[4])
                batch_label_h_matrices.append(sample[5])
                batch_label_w_matrices.append(sample[6])
                label_seq = [each[0] for each in sample[2]]
                batch_label_seqs.append(torch.LongTensor(label_seq).to(args.device))

                input_seq_embed = input_traj_to_embeddings(sample, args.embed_mode, 'eval', freeze=freeze,
                                                           poi_sage_embeddings=poi_sage_embeddings)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(label_seq))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_label_seqs, batch_first=True, padding_value=-1)
            batch_user_embedding = user_embed_model(torch.LongTensor(batch_user).to(args.device))
            if not args.pure_transformer:
                batch_input_h_matrices = torch.stack(batch_input_h_matrices).to(args.device)
                batch_input_w_matrices = torch.stack(batch_input_w_matrices).to(args.device)
                batch_label_h_matrices = torch.stack(batch_label_h_matrices).to(args.device)
                batch_label_w_matrices = torch.stack(batch_label_w_matrices).to(args.device)
            # Feedforward
            x = batch_padded.to(device=args.device)
            y_poi = label_padded_poi.to(device=args.device)
            y_pred_poi = seq_model(x, batch_seq_lens, batch_input_h_matrices, batch_input_w_matrices, batch_label_h_matrices,
                                   batch_label_w_matrices, batch_user_embedding)

            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss = loss_poi

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (10)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)
        if epoch_val_top1_acc > 0.2:
            freeze_set = True
            logging.info("freeze sage embedding")
        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)
    print('ok! it is over.')
    if args.embed_mode!='poi':
        for manager in managers:
            manager.shutdown()
        for p in process_list:
            p.terminate()


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    train(args)
