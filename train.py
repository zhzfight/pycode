import collections
import logging
import logging
import math
import os
import pathlib
import pickle
import random
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
    GraphSAGE, TransformerModel, IntervalAwareTransformer, GRU
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, adj_list, split_list, random_walk_with_restart, \
    get_all_nodes_neighbors, compute_relative_time_matrix, evaluate
import threading


def train(args):
    print("damn!!!\n")
    seq_mode = {'pureTransformer', 'timeIntervalAwareTransformer', 'GRU'}
    if args.seq_mode not in seq_mode:
        print('no sequence model is specified')
        exit(0)
    embed_mode = {'poi', 'sage'}
    if args.embed_mode not in embed_mode:
        print('no embed model is specified')
        exit(0)

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
    df['POI_id'] = pd.factorize(df['POI_id'])[0]+1
    df['user_id'] = pd.factorize(df['user_id'])[0]+1
    df['date_time'] = pd.to_datetime(df['date_time'])
    poi_num = len(set(df['POI_id'].to_list()))

    user_num = len(set(df['user_id'].to_list()))

    # %% ====================== Define Dataset ======================

    class produceSampleProcess(multiprocessing.Process):
        def __init__(self, tasks, node_dicts, adj_list, restart_prob, num_walks, threshold, adjOrdis, id):
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
                        if self.count_dict[node] + threshold > 2000:
                            self.count_dict[node] = 2000
                        else:
                            self.count_dict[node] += threshold

    pois_in_train = set()
    pois_visited_by_user = collections.defaultdict(set)

    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, df, time_threshold, max_len):
            self.df = df.copy()
            self.users = []
            self.seq_lens = []
            self.input_seqs = []
            self.input_timebins = []
            self.input_time_idx = []
            self.dt_for_make_graph=[]
            self.seq_for_make_graph=[]
            self.label_seqs = []
            self.label_timebins = []
            self.label_time_idx = []
            self.input_seq_w_timeMatrixes = []
            self.input_seq_d_timeMatrixes = []
            self.label_seq_w_timeMatrixes = []
            self.label_seq_d_timeMatrixes = []
            self.time_threshold = time_threshold

            for user in tqdm(set(df['user_id'].tolist())):
                user_df = df[df['user_id'] == user]
                user_df = user_df.sort_values(by='date_time')
                poi_ids = user_df['POI_id'].to_list()
                time_bins = user_df['hour_of_week'].to_list()
                d = user_df['hour_of_day'].to_list()
                w = user_df['day_of_week'].to_list()
                dt = user_df['date_time'].to_list()
                self.dt_for_make_graph.append(dt[:-1])
                self.seq_for_make_graph.append(poi_ids[:-1])
                idx = [int(((each - dt[0]) / pd.Timedelta(minutes=15))) for each in dt]
                self.users.append(user)
                self.seq_lens.append(len(poi_ids) - 2)
                paded_len = (max_len - 2) - (len(poi_ids) - 2)
                input_seq = poi_ids[:-2]

                input_seq.extend([0] * paded_len)
                input_timebin = time_bins[:-2]
                input_timebin.extend([0] * paded_len)
                input_time_idx = idx[:-2]
                input_time_idx.extend([99999] * paded_len)
                self.input_seqs.append(input_seq)
                self.input_timebins.append(input_timebin)
                self.input_time_idx.append(input_time_idx)
                input_d_matrix = compute_relative_time_matrix(d[:-2],
                                                              d[:-2], 24, (max_len - 2))
                input_w_matrix = compute_relative_time_matrix(w[:-2],
                                                              w[:-2], 7, (max_len - 2))
                self.input_seq_d_timeMatrixes.append(input_d_matrix)
                self.input_seq_w_timeMatrixes.append(input_w_matrix)
                label_seq = poi_ids[1:-1]
                label_seq.extend([0] * paded_len)
                label_timebin = time_bins[1:-1]
                label_timebin.extend([0] * paded_len)
                label_time_idx = idx[1:-1]
                label_time_idx.extend([99999] * paded_len)
                self.label_seqs.append(label_seq)
                self.label_timebins.append(label_timebin)
                self.label_time_idx.append(label_time_idx)
                label_d_matrix = compute_relative_time_matrix(d[1:-1],
                                                              d[1:-1], 24, (max_len - 2))
                label_w_matrix = compute_relative_time_matrix(w[1:-1],
                                                              w[1:-1], 7, (max_len - 2))
                self.label_seq_d_timeMatrixes.append(label_d_matrix)
                self.label_seq_w_timeMatrixes.append(label_w_matrix)

                pois_in_train.update(poi_ids[:-1])
                pois_visited_by_user[user].update(poi_ids[:-1])

        def get_adj(self):
            adj = collections.defaultdict(dict)
            for datetime,seq in zip(self.dt_for_make_graph,self.seq_for_make_graph):
                pois = [each for each in seq]
                dt = [each for each in datetime]
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

        def get_node_geo(self):
            df = self.df.drop_duplicates(subset='POI_id')
            # 根据poi_set中的poiid从df中筛选出对应的行
            selected_rows = df[df['POI_id'].isin(pois_in_train)]
            # 取出这些行中的lat和long列
            pois = selected_rows['POI_id'].to_list()
            geos = list(zip(selected_rows['longitude'].to_list(), selected_rows['latitude'].to_list()))
            return pois, geos

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (
                self.users[index], self.seq_lens[index], self.input_seqs[index], self.input_timebins[index],
                self.input_time_idx[index],
                self.label_seqs[index], self.label_timebins[index], self.label_time_idx[index],
                self.input_seq_d_timeMatrixes[index],
                self.input_seq_w_timeMatrixes[index], self.label_seq_d_timeMatrixes[index],
                self.label_seq_w_timeMatrixes[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df, max_len):
            self.users = []

            self.input_seqs = []
            self.seq_lens = []
            self.input_timebins = []
            self.input_time_idx = []
            self.label_seqs = []
            self.label_timebins = []
            self.label_time_idx = []
            self.input_seq_w_timeMatrixes = []
            self.input_seq_d_timeMatrixes = []
            self.label_seq_w_timeMatrixes = []
            self.label_seq_d_timeMatrixes = []

            for user in tqdm(set(df['user_id'].tolist())):

                user_df = df[df['user_id'] == user]
                user_df = user_df.sort_values(by='date_time')

                poi_ids = user_df['POI_id'].to_list()
                time_bins = user_df['hour_of_week'].to_list()
                d = user_df['hour_of_day'].to_list()
                w = user_df['day_of_week'].to_list()
                dt = user_df['date_time'].to_list()

                if poi_ids[-1] not in pois_in_train:
                    break

                idx = [int(((each - dt[0]) / pd.Timedelta(minutes=15))) for each in dt]

                self.users.append(user)
                self.seq_lens.append(len(poi_ids) - 1)
                paded_len = (max_len - 1) - (len(poi_ids) - 1)
                input_seq = poi_ids[:-1]
                input_seq.extend([0] * paded_len)
                input_timebin = time_bins[:-1]
                input_timebin.extend([0] * paded_len)
                input_time_idx = idx[:-1]
                input_time_idx.extend([99999] * paded_len)
                self.input_seqs.append(input_seq)
                self.input_timebins.append(input_timebin)
                self.input_time_idx.append(input_time_idx)

                input_d_matrix = compute_relative_time_matrix(d[:-1],
                                                              d[:-1], 24, (max_len - 1))
                input_w_matrix = compute_relative_time_matrix(w[:-1],
                                                              w[:-1], 7, (max_len - 1))
                self.input_seq_d_timeMatrixes.append(input_d_matrix)
                self.input_seq_w_timeMatrixes.append(input_w_matrix)
                label_seq = poi_ids[1:]
                label_seq.extend([0] * paded_len)
                label_timebin = time_bins[1:]
                label_timebin.extend([0] * paded_len)
                label_time_idx = idx[1:]
                label_time_idx.extend([99999] * paded_len)
                self.label_seqs.append(label_seq)
                self.label_timebins.append(label_timebin)
                self.label_time_idx.append(label_time_idx)

                label_d_matrix = compute_relative_time_matrix(d[1:],
                                                              d[1:], 24, (max_len - 1))
                label_w_matrix = compute_relative_time_matrix(w[1:],
                                                              w[1:], 7, (max_len - 1))
                self.label_seq_d_timeMatrixes.append(label_d_matrix)
                self.label_seq_w_timeMatrixes.append(label_w_matrix)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (
                self.users[index], self.seq_lens[index], self.input_seqs[index], self.input_timebins[index],
                self.input_time_idx[index],
                self.label_seqs[index], self.label_timebins[index], self.label_time_idx[index],
                self.input_seq_d_timeMatrixes[index],
                self.input_seq_w_timeMatrixes[index], self.label_seq_d_timeMatrixes[index],
                self.label_seq_w_timeMatrixes[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(df, pd.Timedelta(6, unit='h'),args.max_len)
    val_dataset = TrajectoryDatasetVal(df,args.max_len)
    pois_not_visited_by_user = collections.defaultdict(list)
    for k, v in pois_visited_by_user.items():
        pois_not_visited_by_user[k] = list(pois_in_train.difference(v))

    def time_collate_fn(batch):
        users, seq_lens,input_seqs, input_timebins, input_time_idx, label_seqs, label_timebins, \
            label_time_idx, input_seq_d_matrices, input_seq_w_matrices, \
            label_seq_d_matrices, label_seq_w_matrices = map(list, zip(*batch))

        def process_list(user, lst):
            neg = random.sample(pois_not_visited_by_user[user], len(lst))
            return neg

        input_seq_d_matrices=np.array(input_seq_d_matrices)
        input_seq_w_matrices=np.array(input_seq_w_matrices)
        label_seq_d_matrices=np.array(label_seq_d_matrices)
        label_seq_w_matrices=np.array(label_seq_w_matrices)
        # 对元组中的每个列表应用自定义函数
        negs = [process_list(user, lst) for user, lst in zip(users, label_seqs)]
        return users,seq_lens, input_seqs, input_timebins, input_time_idx, label_seqs, label_timebins, \
            label_time_idx, input_seq_d_matrices, input_seq_w_matrices, \
            label_seq_d_matrices, label_seq_w_matrices, negs

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=time_collate_fn)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=time_collate_fn)

    adj = None
    dis = None
    if args.embed_mode == 'sage':
        basename = os.path.basename(args.dataset)
        prefix, _ = os.path.splitext(basename)
        if os.path.exists(os.path.join(os.path.dirname(args.dataset), prefix + '_adj.pkl')):
            with open(os.path.join(os.path.dirname(args.dataset), prefix + '_adj.pkl'), 'rb') as f:  # 打开pickle文件
                adj = pickle.load(f)  # 读取字典
        else:
            adj = train_dataset.get_adj()
            with open(os.path.join(os.path.dirname(args.dataset), prefix + '_adj.pkl'), 'wb') as f:
                pickle.dump(adj, f)  # 把字典写入pickle文件

        if os.path.exists(os.path.join(os.path.dirname(args.dataset), prefix + '_dis.pkl')):
            with open(os.path.join(os.path.dirname(args.dataset), prefix + '_dis.pkl'), 'rb') as f:  # 打开pickle文件
                dis = pickle.load(f)  # 读取字典
        else:
            pois, geos = train_dataset.get_node_geo()
            print('space neighbor table making, if you have multi cpus, it will be faster.')
            dis = get_all_nodes_neighbors(pois, geos, args.geo_k, args.geo_dis)
            with open(os.path.join(os.path.dirname(args.dataset), prefix + '_dis.pkl'), 'wb') as f:
                pickle.dump(dis, f)  # 把字典写入pickle文件
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
    managers = None

    threshold = 10  # 队列大小阈值
    process_list = []
    if args.embed_mode == 'sage':
        managers = []
        managers_num = 100
        for _ in range(managers_num):
            managers.append(multiprocessing.Manager())
        adj_dicts = [managers[i % managers_num].Queue() for i in range(poi_num)]
        dis_dicts = [managers[i % managers_num].Queue() for i in range(poi_num)]
        tasks = split_list([i for i in range(poi_num)], int(args.cpus / 2))
        for idx, task in enumerate(tasks):
            ap = produceSampleProcess(tasks=task, node_dicts=adj_dicts, adj_list=adj, restart_prob=args.restart_prob,
                                      num_walks=args.num_walks,
                                      threshold=threshold, adjOrdis='adj', id=idx)
            ap.start()
            process_list.append(ap)
            dp = produceSampleProcess(tasks=task, node_dicts=dis_dicts, adj_list=dis, restart_prob=args.restart_prob,
                                      num_walks=args.num_walks,
                                      threshold=threshold, adjOrdis='dis', id=idx)
            dp.start()
            process_list.append(dp)

    # %% ====================== Build Models ======================
    poi_embed_model = None
    if args.embed_mode == 'sage':
        poi_embed_model = GraphSAGE(poi_num=poi_num, input_dim=args.poi_sage_dim, embed_dim=args.poi_sage_dim,
                                    device=args.device, restart_prob=args.restart_prob, num_walks=args.num_walks,
                                    dropout=args.dropout, adj_dicts=adj_dicts, dis_dicts=dis_dicts, adj_list=adj,
                                    dis_list=dis)
    elif args.embed_mode == 'poi':
        poi_embed_model = PoiEmbeddings(poi_num, args.poi_id_dim)

    # %% Model3: Time Model
    time_embed_model = None
    if args.use_time_feat:
        time_embed_model = TimeEmbeddings(args.time_embed_dim)


    # %% Model6: Sequence model
    poi_embed_dim = 0
    if args.embed_mode == 'sage':
        poi_embed_dim = args.poi_sage_dim
    elif args.embed_mode == 'poi':
        poi_embed_dim = args.poi_id_dim
    args.seq_input_embed = poi_embed_dim

    if args.use_time_feat:
        args.seq_input_embed += args.time_embed_dim
    user_embed_model = None
    if args.seq_mode == 'timeIntervalAwareTransformer':
        user_embed_model = UserEmbeddings(user_num, args.seq_input_embed)
    seq_model = None
    if args.seq_mode == 'pureTransformer':
        seq_model = TransformerModel(poi_num,
                                     args.seq_input_embed,
                                     nhead=1,
                                     nhid=args.seq_input_embed,
                                     nlayers=args.nlayer,
                                     device=args.device,
                                     dropout=args.dropout)
    elif args.seq_mode == 'timeIntervalAwareTransformer':
        seq_model = IntervalAwareTransformer(num_poi=poi_num,
                                             nhid=args.seq_input_embed,
                                             batch_size=args.batch,
                                             device=args.device,
                                             dropout=args.dropout,
                                             max_len=args.max_len, nlayer=args.nlayer)
    elif args.seq_mode == 'GRU':
        seq_model = GRU(nhid=args.seq_input_embed, num_poi=poi_num)

    # Define overall loss and optimizer
    parameter_list = list(seq_model.parameters()) + list(poi_embed_model.parameters())
    if args.seq_mode == 'timeIntervalAwareTransformer':
        parameter_list += list(user_embed_model.parameters())


    if args.use_time_feat:
        parameter_list += list(time_embed_model.parameters())
    optimizer = optim.Adam(params=parameter_list,
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)


    # %% ====================== Train ======================

    poi_embed_model = poi_embed_model.to(device=args.device)
    if args.seq_mode == 'timeIntervalAwareTransformer':
        user_embed_model = user_embed_model.to(device=args.device)
    if args.use_time_feat:
        time_embed_model = time_embed_model.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_ndcg_list = []
    train_epochs_hit_list = []
    train_epochs_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_ndcg_list = []
    val_epochs_hit_list = []
    val_epochs_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")

        poi_embed_model.train()
        if args.seq_mode == 'timeIntervalAwareTransformer':
            user_embed_model.train()
        if args.use_time_feat:
            time_embed_model.train()
        seq_model.train()
        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_ndcg_list = []
        train_batches_hit_list = []


        for b_idx, (users,seq_lens, input_seqs, input_timebins, input_time_idxes, label_seqs, label_timebins, \
                label_time_idxes, input_seq_d_matrices, input_seq_w_matrices, \
                label_seq_d_matrices, label_seq_w_matrices, negs) in enumerate(train_loader):

            x=poi_embed_model(torch.LongTensor(input_seqs).to(args.device),seq_lens)
            if args.use_time_feat:
                timebin_embed = time_embed_model(torch.LongTensor(input_timebins).to(args.device))
                x = torch.cat((x, timebin_embed), dim=-1)

            user_embeddings = None
            if args.seq_mode == 'timeIntervalAwareTransformer':
                user_embeddings = user_embed_model(torch.LongTensor(users).to(args.device))

            input_time_idxes = torch.FloatTensor(input_time_idxes).to(args.device)
            label_time_idxes = torch.FloatTensor(label_time_idxes).to(args.device)
            input_seq_d_matrices = torch.LongTensor(input_seq_d_matrices).to(args.device)
            input_seq_w_matrices = torch.LongTensor(input_seq_w_matrices).to(args.device)
            label_seq_d_matrices = torch.LongTensor(label_seq_d_matrices).to(args.device)
            label_seq_w_matrices = torch.LongTensor(label_seq_w_matrices).to(args.device)


            preference = seq_model(x, seq_lens, input_seq_d_matrices, input_seq_w_matrices,
                                   label_seq_d_matrices,
                                   label_seq_w_matrices, user_embeddings, input_time_idxes,
                                   label_time_idxes)
            label_seqs_for_loss=torch.LongTensor(label_seqs).to(args.device)
            label_seqs_for_loss-=1
            loss = criterion_poi(preference.transpose(1, 2), label_seqs_for_loss)

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
            batch_label_pois = label_seqs_for_loss.detach().cpu().numpy()
            batch_pred_pois = preference.detach().cpu().numpy()

            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, seq_lens):
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

            ndcg_list, hit_list = evaluate(batch_pred_pois, batch_label_pois, seq_lens, 10)
            batch_ndcg = np.mean(ndcg_list)
            batch_hit = np.mean(hit_list)
            train_batches_ndcg_list.append(batch_ndcg)
            train_batches_hit_list.append(batch_hit)
            train_batches_loss_list.append(loss.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (10)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_ndcg:{batch_ndcg:.4f}\n'
                             f'train_hit:{batch_hit:.4f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{users[sample_idx]}\n'
                             f'input_seq: {input_seqs[sample_idx][:seq_lens[sample_idx]]}\n'
                             f'label_seq:{label_seqs[sample_idx][:seq_lens[sample_idx]]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:seq_lens[sample_idx]])} \n' +
                             '=' * 100)
        if epoch%5!=0:
            continue
        # train end --------------------------------------------------------------------------------------------------------

        poi_embed_model.eval()
        if args.seq_mode == 'timeIntervalAwareTransformer':
            user_embed_model.eval()
        if args.use_time_feat:
            time_embed_model.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_ndcg_list = []
        val_batches_hit_list = []


        embed_table=None
        if args.embed_mode == 'sage':
            embed_table = poi_embed_model(None)
        elif args.embed_mode=='poi':
            embed_table=poi_embed_model.weight()
        for v_idx, (users,seq_lens, input_seqs, input_timebins, input_time_idxes, label_seqs, label_timebins, \
                label_time_idxes, input_seq_d_matrices, input_seq_w_matrices, \
                label_seq_d_matrices, label_seq_w_matrices, negs) in enumerate(val_loader):

            x=embed_table[input_seqs]

            if args.use_time_feat:
                timebin_embed = time_embed_model(torch.LongTensor(input_timebins).to(args.device))
                x = torch.cat((x, timebin_embed), dim=-1)
            user_embeddings = None
            if args.seq_mode == 'timeIntervalAwareTransformer':
                user_embeddings = user_embed_model(torch.LongTensor(users).to(args.device))

            input_time_idxes = torch.FloatTensor(input_time_idxes).to(args.device)
            label_time_idxes = torch.FloatTensor(label_time_idxes).to(args.device)
            input_seq_d_matrices = torch.LongTensor(input_seq_d_matrices).to(args.device)
            input_seq_w_matrices = torch.LongTensor(input_seq_w_matrices).to(args.device)
            label_seq_d_matrices = torch.LongTensor(label_seq_d_matrices).to(args.device)
            label_seq_w_matrices = torch.LongTensor(label_seq_w_matrices).to(args.device)


            label_seqs_for_loss=torch.LongTensor(label_seqs).to(args.device)
            label_seqs_for_loss-=1
            preference = seq_model(x, seq_lens, input_seq_d_matrices, input_seq_w_matrices,
                                   label_seq_d_matrices,
                                   label_seq_w_matrices, user_embeddings, input_time_idxes,
                                   label_time_idxes)

            # Calculate loss
            loss = criterion_poi(preference.transpose(1, 2),label_seqs_for_loss)
            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0

            batch_label_pois = label_seqs_for_loss.detach().cpu().numpy()
            batch_pred_pois = preference.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, seq_lens):
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

            ndcg_list, hit_list = evaluate(batch_pred_pois, batch_label_pois, seq_lens, 10)
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            batch_ndcg = np.mean(ndcg_list)
            batch_hit = np.mean(hit_list)
            val_batches_ndcg_list.append(batch_ndcg)
            val_batches_hit_list.append(batch_hit)

            # Report validation progress
            if (v_idx % (10)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{v_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_ndcg:{batch_ndcg:.4f}\n'
                             f'val_hit:{batch_hit:.4f}\n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{users[sample_idx]}\n'
                             f'input_seq:{input_seqs[sample_idx][:seq_lens[sample_idx]]}\n'
                             f'label_seq:{label_seqs[sample_idx][:seq_lens[sample_idx]]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:seq_lens[sample_idx]])} \n' +
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
        epoch_train_ndcg = np.mean(train_batches_ndcg_list)
        epoch_train_hit = np.mean(train_batches_hit_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_ndcg = np.mean(val_batches_ndcg_list)
        epoch_val_hit = np.mean(val_batches_hit_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_ndcg_list.append(epoch_train_ndcg)
        train_epochs_hit_list.append(epoch_train_hit)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)

        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_ndcg_list.append(epoch_val_ndcg)
        val_epochs_hit_list.append(epoch_val_hit)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)


        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_ndcg:{epoch_train_ndcg:.4f}, "
                     f"train_hit:{epoch_train_hit:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_ndcg: {epoch_val_ndcg:.4f}, "
                     f"val_hit: {epoch_val_hit:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}"
                     )

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_ndcg_list={[float(f"{each:.4f}") for each in train_epochs_ndcg_list]}', file=f)
            print(f'train_epochs_hit_list={[float(f"{each:.4f}") for each in train_epochs_hit_list]}', file=f)
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
            print(f'val_epochs_ndcg_list={[float(f"{each:.4f}") for each in val_epochs_ndcg_list]}', file=f)
            print(f'val_epochs_hit_list={[float(f"{each:.4f}") for each in val_epochs_hit_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)
    print('ok! it is over.')
    if args.embed_mode != 'poi':
        for manager in managers:
            manager.shutdown()
        for p in process_list:
            p.terminate()


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    train(args)
