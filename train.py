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
    GraphSAGE, TransformerModel, IntervalAwareTransformer, GRU
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, adj_list, split_list, random_walk_with_restart, \
    get_all_nodes_neighbors, compute_relative_time_matrix, evaluate
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
    df['POI_id'] = pd.factorize(df['POI_id'])[0]
    df['user_id'] = pd.factorize(df['user_id'])[0]
    df['date_time'] = pd.to_datetime(df['date_time'])
    poi_num = len(set(df['POI_id'].to_list()))
    if args.use_cat_feat:
        df['POI_catid'] = pd.factorize(df['POI_catid'])[0]
        cat_num = len(set(df['POI_catid'].to_list()))
        poi2cat = df.set_index('POI_id')['POI_catid'].to_dict()

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
            self.time_threshold = time_threshold
            for user in tqdm(set(df['user_id'].tolist())):
                user_df = df[df['user_id'] == user]
                user_df = user_df.sort_values(by='date_time')
                poi_ids = user_df['POI_id'].to_list()
                time_bins = user_df['hour_of_week'].to_list()
                h = user_df['hour_of_day'].to_list()
                w = user_df['day_of_week'].to_list()
                dt = user_df['date_time'].to_list()
                idx = [int(((each - dt[0]) / pd.Timedelta(minutes=15))) for each in dt]
                self.users.append(user)
                self.input_seqs.append(list(
                    zip(poi_ids[:-2], time_bins[:-2],
                        h[:-2], w[:-2],
                        dt[:-2], idx[:-2])))
                input_h_matrix = compute_relative_time_matrix(h[:-2],
                                                              h[:-2], 24)
                input_w_matrix = compute_relative_time_matrix(w[:-2],
                                                              w[:-2], 7)
                self.input_seq_h_timeMatrixes.append(torch.LongTensor(input_h_matrix))
                self.input_seq_w_timeMatrixes.append(torch.LongTensor(input_w_matrix))
                self.label_seqs.append(list(
                    zip(poi_ids[1:-1], time_bins[1:-1],
                        h[1:-1], w[1:-1],
                        dt[1:-1], idx[1:-1])))
                label_h_matrix = compute_relative_time_matrix(h[1:-1],
                                                              h[1:-1], 24)
                label_w_matrix = compute_relative_time_matrix(w[1:-1],
                                                              w[1:-1], 7)
                self.label_seq_h_timeMatrixes.append(torch.LongTensor(label_h_matrix))
                self.label_seq_w_timeMatrixes.append(torch.LongTensor(label_w_matrix))

                pois_in_train.update(poi_ids[:-2])

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

        def get_node_geo(self):
            df = self.df.drop_duplicates(subset='POI_id')
            # 根据poi_set中的poiid从df中筛选出对应的行
            selected_rows = df[df['POI_id'].isin(pois_in_train)]
            # 取出这些行中的lat和long列
            pois= selected_rows['POI_id'].to_list()
            geos = list(zip(selected_rows['longitude'].to_list(),selected_rows['latitude'].to_list()))
            return pois, geos

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

            for user in tqdm(set(df['user_id'].tolist())):

                user_df = df[df['user_id'] == user]
                user_df = user_df.sort_values(by='date_time')

                poi_ids = user_df['POI_id'].to_list()
                time_bins = user_df['hour_of_week'].to_list()
                h = user_df['hour_of_day'].to_list()
                w = user_df['day_of_week'].to_list()
                dt = user_df['date_time'].to_list()


                if poi_ids[-1] not in pois_in_train:
                    break

                idx = [int(((each - dt[0]) / pd.Timedelta(minutes=15))) for each in dt]
                self.users.append(user)
                self.input_seqs.append(list(
                    zip(poi_ids[:-1], time_bins[:-1],
                        h[:-1], w[:-1],
                        idx[:-1])))
                input_h_matrix = compute_relative_time_matrix(h[:-1],
                                                              h[:-1], 24)
                input_w_matrix = compute_relative_time_matrix(w[:-1],
                                                              w[:-1], 7)
                self.input_seq_h_timeMatrixes.append(torch.LongTensor(input_h_matrix))
                self.input_seq_w_timeMatrixes.append(torch.LongTensor(input_w_matrix))
                self.label_seqs.append(list(
                    zip(poi_ids[1:], time_bins[1:],
                        h[1:], w[1:],
                        idx[1:])))
                label_h_matrix = compute_relative_time_matrix(h[1:],
                                                              h[1:], 24)
                label_w_matrix = compute_relative_time_matrix(w[1:],
                                                              w[1:], 7)
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

    collate_fn = lambda x: x
    if args.seq_mode == 'timeIntervalAwareTransformer':
        collate_fn = time_collate_fn
    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(df, pd.Timedelta(6, unit='h'))
    val_dataset = TrajectoryDatasetVal(df)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=collate_fn)

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

    # %% Model4: Category embedding model
    cat_embed_model = None
    if args.use_cat_feat:
        cat_embed_model = CategoryEmbeddings(cat_num, args.cat_embed_dim)

    # %% Model6: Sequence model
    poi_embed_dim = 0
    if args.embed_mode == 'sage':
        poi_embed_dim = args.poi_sage_dim
    elif args.embed_mode == 'poi':
        poi_embed_dim = args.poi_id_dim
    args.seq_input_embed = poi_embed_dim
    if args.use_cat_feat:
        args.seq_input_embed += args.cat_embed_dim
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
                                             max_len=args.max_seq_len, nlayer=args.nlayer)
    elif args.seq_mode == 'GRU':
        seq_model = GRU(nhid=args.seq_input_embed, num_poi=poi_num)

    # Define overall loss and optimizer
    parameter_list = list(seq_model.parameters()) + list(poi_embed_model.parameters())
    if args.seq_mode == 'timeIntervalAwareTransformer':
        parameter_list += list(user_embed_model.parameters())

    if args.use_cat_feat:
        parameter_list += list(cat_embed_model.parameters())

    if args.use_time_feat:
        parameter_list += list(time_embed_model.parameters())
    optimizer = optim.Adam(params=parameter_list,
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, mode, train_or_eval, use_cat_feat=False,
                                 use_time_feat=False, poi_sage_embeddings=None,
                                 embedding_index=None):
        input_seq = [each[0] for each in sample[1]]
        input_seq_cat = None
        input_seq_time = None
        if use_cat_feat:
            input_seq_cat = [poi2cat[each] for each in input_seq]
        if use_time_feat:
            input_seq_time = [each[1] for each in sample[1]]
        if mode == 'sage':
            if train_or_eval == 'eval':
                poi_idxs = input_seq
            else:
                poi_idxs = [embedding_index + idx for idx in range(len(input_seq))]
            poi_embed = poi_sage_embeddings[poi_idxs]
        else:
            poi_embed = poi_embed_model(torch.LongTensor(input_seq).to(args.device))

        embedded = poi_embed
        if use_cat_feat:
            catid_embed = cat_embed_model(torch.LongTensor(input_seq_cat).to(args.device))
            embedded = torch.cat((embedded, catid_embed), dim=-1)
        if use_time_feat:
            timebin_embed = time_embed_model(torch.LongTensor(input_seq_time).to(args.device))
            embedded = torch.cat((embedded, timebin_embed), dim=-1)

        return embedded

    # %% ====================== Train ======================

    poi_embed_model = poi_embed_model.to(device=args.device)
    if args.seq_mode == 'timeIntervalAwareTransformer':
        user_embed_model = user_embed_model.to(device=args.device)
    if args.use_time_feat:
        time_embed_model = time_embed_model.to(device=args.device)
    if args.use_cat_feat:
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
        if args.use_cat_feat:
            cat_embed_model.train()
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

        poi_sage_embeddings = None
        embedding_index = 0

        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            # For padding

            batch_seq_lens = []
            batch_seq_embeds = []
            batch_label_seqs = []
            batch_user = []
            batch_input_h_matrices = []
            batch_input_w_matrices = []
            batch_label_h_matrices = []
            batch_label_w_matrices = []
            batch_input_idxes = []
            batch_label_idxes = []
            if args.embed_mode == 'sage':
                pois = [each[0] for sample in batch for each in sample[1]]
                poi_sage_embeddings = poi_embed_model(torch.tensor(pois).to(args.device))
            # Convert input seq to embeddings
            for sample in batch:
                batch_user.append(sample[0])
                batch_input_h_matrices.append(sample[3])
                batch_input_w_matrices.append(sample[4])
                batch_label_h_matrices.append(sample[5])
                batch_label_w_matrices.append(sample[6])
                label_seq = [each[0] for each in sample[2]]
                input_seq_idx = [each[5] for each in sample[1]]
                label_seq_idx = [each[5] for each in sample[2]]
                batch_input_idxes.append(torch.FloatTensor(input_seq_idx).to(args.device))
                batch_label_idxes.append(torch.FloatTensor(label_seq_idx).to(args.device))
                batch_label_seqs.append(torch.LongTensor(label_seq).to(args.device))

                input_seq_embed = input_traj_to_embeddings(sample, args.embed_mode, 'train',
                                                           use_cat_feat=args.use_cat_feat,
                                                           use_time_feat=args.use_time_feat,
                                                           poi_sage_embeddings=poi_sage_embeddings,
                                                           embedding_index=embedding_index)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(label_seq))
                embedding_index += len(label_seq)
            embedding_index = 0
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_label_seqs, batch_first=True, padding_value=-1)
            batch_padded_input_idxes = pad_sequence(batch_input_idxes, batch_first=True, padding_value=99999).to(
                args.device)
            batch_padded_label_idxes = pad_sequence(batch_label_idxes, batch_first=True, padding_value=99999).to(
                args.device)
            batch_user_embedding = None
            if args.seq_mode == 'timeIntervalAwareTransformer':
                batch_user_embedding = user_embed_model(torch.LongTensor(batch_user).to(args.device))

            # Feedforward
            if args.seq_mode == 'timeIntervalAwareTransformer':
                batch_input_h_matrices = torch.stack(batch_input_h_matrices).to(args.device)
                batch_input_w_matrices = torch.stack(batch_input_w_matrices).to(args.device)
                batch_label_h_matrices = torch.stack(batch_label_h_matrices).to(args.device)
                batch_label_w_matrices = torch.stack(batch_label_w_matrices).to(args.device)

            x = batch_padded.to(device=args.device)
            y_poi = label_padded_poi.to(device=args.device)
            y_pred_poi = seq_model(x, batch_seq_lens, batch_input_h_matrices, batch_input_w_matrices,
                                   batch_label_h_matrices,
                                   batch_label_w_matrices, batch_user_embedding, batch_padded_input_idxes,
                                   batch_padded_label_idxes)

            loss = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)

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

            ndcg_list, hit_list = evaluate(batch_pred_pois, batch_label_pois, batch_seq_lens, 10)
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
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)

        # train end --------------------------------------------------------------------------------------------------------

        poi_embed_model.eval()
        if args.seq_mode == 'timeIntervalAwareTransformer':
            user_embed_model.eval()
        if args.use_time_feat:
            time_embed_model.eval()
        if args.use_cat_feat:
            cat_embed_model.eval()
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

        poi_sage_embeddings = None

        if args.embed_mode == 'sage':
            pois = [n for n in range(poi_num)]
            poi_sage_embeddings = poi_embed_model(torch.tensor(pois).to(args.device))
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
            batch_input_idxes = []
            batch_label_idxes = []
            # Convert input seq to embeddings
            for sample in batch:
                batch_user.append(sample[0])
                batch_input_h_matrices.append(sample[3])
                batch_input_w_matrices.append(sample[4])
                batch_label_h_matrices.append(sample[5])
                batch_label_w_matrices.append(sample[6])
                label_seq = [each[0] for each in sample[2]]
                input_seq_idx = [each[4] for each in sample[1]]
                label_seq_idx = [each[4] for each in sample[2]]
                batch_input_idxes.append(torch.FloatTensor(input_seq_idx).to(args.device))
                batch_label_idxes.append(torch.FloatTensor(label_seq_idx).to(args.device))
                batch_label_seqs.append(torch.LongTensor(label_seq).to(args.device))

                input_seq_embed = input_traj_to_embeddings(sample, args.embed_mode, 'eval',
                                                           use_cat_feat=args.use_cat_feat,
                                                           use_time_feat=args.use_time_feat,
                                                           poi_sage_embeddings=poi_sage_embeddings)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(label_seq))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_label_seqs, batch_first=True, padding_value=-1)
            batch_padded_input_idxes = pad_sequence(batch_input_idxes, batch_first=True, padding_value=99999).to(
                args.device)
            batch_padded_label_idxes = pad_sequence(batch_label_idxes, batch_first=True, padding_value=99999).to(
                args.device)
            batch_user_embedding = None
            if args.seq_mode == 'timeIntervalAwareTransformer':
                batch_user_embedding = user_embed_model(torch.LongTensor(batch_user).to(args.device))
            if args.seq_mode == 'timeIntervalAwareTransformer':
                batch_input_h_matrices = torch.stack(batch_input_h_matrices).to(args.device)
                batch_input_w_matrices = torch.stack(batch_input_w_matrices).to(args.device)
                batch_label_h_matrices = torch.stack(batch_label_h_matrices).to(args.device)
                batch_label_w_matrices = torch.stack(batch_label_w_matrices).to(args.device)
            # Feedforward
            x = batch_padded.to(device=args.device)
            y_poi = label_padded_poi.to(device=args.device)
            y_pred_poi = seq_model(x, batch_seq_lens, batch_input_h_matrices, batch_input_w_matrices,
                                   batch_label_h_matrices,
                                   batch_label_w_matrices, batch_user_embedding, batch_padded_input_idxes,
                                   batch_padded_label_idxes)

            # Calculate loss
            loss = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
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

            ndcg_list, hit_list = evaluate(batch_pred_pois, batch_label_pois, batch_seq_lens, 10)
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            batch_ndcg = np.mean(ndcg_list)
            batch_hit = np.mean(hit_list)
            val_batches_ndcg_list.append(batch_ndcg)
            val_batches_hit_list.append(batch_hit)

            # Report validation progress
            if (vb_idx % (10)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
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

        # Monitor loss and score
        monitor_loss = epoch_val_loss

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)
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
