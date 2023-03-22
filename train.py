import logging
import logging
import os
import pathlib
import pickle
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import  UserEmbeddings, CategoryEmbeddings, FuseEmbeddings, TransformerModel, \
    PoiEmbeddings, TimeEmbeddings, CAPE, TimeIntervalEmbeddings
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, data_preprocessing, split_df, \
    computeTimeIntervalMatrix


def train(args):
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

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    df, num_pois, num_cats, num_users, max_len, mean_len = data_preprocessing(args.path, args.drop_poi, args.drop_user, \
                                                                              args.split_time, args.seqlen_thre)

    train_df, test_df = split_df(df)
    logging.info(f"num_pois:{num_pois}, num_cats:{num_cats}, max_len:{max_len}, mean_len:{mean_len}")
    print(f"num_pois:{num_pois}, num_cats:{num_cats}, max_len:{max_len}, mean_len:{mean_len}")
    # %% ====================== Define Dataset ======================
    train_user_set = set()

    def caculatTimeMatrix(date_time, max_len, a):
        matrix = [[0] * max_len] * max_len
        for i in range(len(date_time)):
            for j in range(len(date_time)):
                matrix[i][j + a] = abs((date_time[i] - date_time[j]).total_seconds())
        return matrix

    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df, max_len):
            self.users = []
            self.input_seqs_poi = []
            self.input_seqs_cat = []
            self.input_seqs_time_feature = []
            self.label_seqs_poi = []
            self.label_seqs_cat = []
            self.label_seqs_timeInterval = []
            self.input_seqs_timeMatrix = []
            self.input_seqlens = []

            for traj_id in tqdm(set(train_df['traj_id'].tolist())):
                user_id = int(traj_id.split('_')[0])
                train_user_set.add(user_id)

                traj_df = train_df[train_df['traj_id'] == traj_id]
                seqlen = len(traj_df) - 1
                poi_ids = traj_df['venueId'].tolist()
                cat_ids = traj_df['venueCategoryId'].tolist()
                time_feature = traj_df['time_feature'].tolist()
                time_feature = [int(x * args.time_slot) for x in time_feature]
                date_time = traj_df['datetime'].tolist()

                a = max(0, max_len - seqlen)
                b = max(0, seqlen - max_len)
                ip = [0] * max_len
                ic = [0] * max_len
                it = [0] * max_len
                lp = [0] * max_len
                lc = [0] * max_len
                lti = [0] * max_len

                ip[a:] = poi_ids[b:-1]
                ic[a:] = cat_ids[b:-1]
                it[a:] = time_feature[b:-1]
                lp[a:] = poi_ids[b + 1:]
                lc[a:] = cat_ids[b + 1:]

                target_Interval = [int((x - y).total_seconds()) for x, y in zip(date_time[1:], date_time[:-1])]
                lti[a:] = target_Interval[b:]

                self.input_seqlens.append(min(seqlen,max_len))
                self.users.append(user_id)
                self.input_seqs_poi.append(ip)
                self.input_seqs_cat.append(ic)
                self.input_seqs_time_feature.append(it)
                self.label_seqs_poi.append(lp)
                self.label_seqs_cat.append(lc)
                self.label_seqs_timeInterval.append(lti)
                self.input_seqs_timeMatrix.append(caculatTimeMatrix(date_time[:-1],max_len,a))

        def __len__(self):
            return len(self.users)

        def __getitem__(self, index):
            return self.users[index], self.input_seqs_poi[index], self.input_seqs_cat[index], \
                self.input_seqs_time_feature[index], \
                self.label_seqs_poi[index], self.label_seqs_cat[index], \
                self.label_seqs_timeInterval[index], self.input_seqs_timeMatrix[index], self.input_seqlens[index]

    class TrajectoryDatasetTest(Dataset):
        def __init__(self, test_df, max_len):

            self.users = []
            self.input_seqs_poi = []
            self.input_seqs_cat = []
            self.input_seqs_time_feature = []
            self.label_seqs_poi = []
            self.label_seqs_cat = []
            self.label_seqs_timeInterval = []
            self.input_seqs_timeMatrix = []
            self.input_seqlens = []

            for traj_id in tqdm(set(test_df['traj_id'].tolist())):
                user_id = int(traj_id.split('_')[0])

                # Ignore user if not in training set
                if user_id not in train_user_set:
                    continue
                traj_df = test_df[test_df['traj_id'] == traj_id]
                seqlen = len(traj_df) - 1
                poi_ids = traj_df['venueId'].tolist()
                cat_ids = traj_df['venueCategoryId'].tolist()
                time_feature = traj_df['time_feature'].tolist()
                time_feature=[int(x*args.time_slot) for x in time_feature]
                date_time = traj_df['datetime'].tolist()

                a = max(0, max_len - seqlen)
                b = max(0, seqlen - max_len)
                ip = [0] * max_len
                ic = [0] * max_len
                it = [0] * max_len
                lp = [0] * max_len
                lc = [0] * max_len
                lti = [0] * max_len

                ip[a:] = poi_ids[b:-1]
                ic[a:] = cat_ids[b:-1]
                it[a:] = time_feature[b:-1]
                lp[a:] = poi_ids[b + 1:]
                lc[a:] = cat_ids[b + 1:]

                target_Interval = [int((x - y).total_seconds()) for x, y in zip(date_time[1:], date_time[:-1])]
                lti[a:] = target_Interval[b:]

                self.input_seqlens.append(min(seqlen,max_len))
                self.users.append(user_id)
                self.input_seqs_poi.append(ip)
                self.input_seqs_cat.append(ic)
                self.input_seqs_time_feature.append(it)
                self.label_seqs_poi.append(lp)
                self.label_seqs_cat.append(lc)
                self.label_seqs_timeInterval.append(lti)
                self.input_seqs_timeMatrix.append(caculatTimeMatrix(date_time[:-1],max_len,a))

        def __len__(self):
            return len(self.users)

        def __getitem__(self, index):
            return self.users[index], self.input_seqs_poi[index], self.input_seqs_cat[index], \
                self.input_seqs_time_feature[index], \
                self.label_seqs_poi[index], self.label_seqs_cat[index], \
                self.label_seqs_timeInterval[index], self.input_seqs_timeMatrix[index], self.input_seqlens[index]

    def process_data(df, context_size):
        # only use the train_df
        p = []
        c = []
        for traj_id in tqdm(set(df['traj_id'].tolist())):
            traj_df = df[df['traj_id'] == traj_id]
            poi_ids = traj_df['venueId'].tolist()
            cat_ids = traj_df['venueCategoryId'].tolist()
            for i in range(len(poi_ids)):
                poi_context = poi_ids[i - context_size:i] + poi_ids[i + 1:i + context_size + 1]
                for poi in poi_context:
                    p.append([poi_ids[i], poi])
                cat_context = cat_ids[i - context_size:i]
                for cat in cat_context:
                    c.append([poi_ids[i], cat_ids[i], cat])
        return p, c

    p, c = process_data(train_df, args.context_size)

    class word2vecPoiContextDataset(Dataset):
        def __init__(self, p):
            self.ps = [x[0] for x in p]
            self.pcs = [x[1] for x in p]

        def __len__(self):
            return len(self.ps)

        def __getitem__(self, index):
            return self.ps[index], self.pcs[index]

    class word2vecCatContextDataset(Dataset):
        def __init__(self, c):
            self.ps = [x[0] for x in c]
            self.cs = [x[1] for x in c]
            self.ccs = [x[2] for x in c]

        def __len__(self):
            return len(self.cs)

        def __getitem__(self, index):
            return self.ps[index], self.cs[index], self.ccs[index]

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df,max_len)
    test_dataset = TrajectoryDatasetTest(test_df,max_len)
    p_dataset = word2vecPoiContextDataset(p)
    c_dataset = word2vecCatContextDataset(c)
    def collect_fn1(batch):
        u = [x[0] for x in batch]
        ip = [x[1] for x in batch]
        ic = [x[2] for x in batch]
        it=[x[3] for x in batch]
        lp=[x[4] for x in batch]
        lc=[x[5] for x in batch]
        lti=[x[6] for x in batch]
        itm=[x[7] for x in batch]
        seqlen=[x[8] for x in batch]
        return u,ip,ic,it,lp,lc,lti,itm,seqlen
    def collect_fn2(batch):
        p=[x[0] for x in batch]
        pc=[x[1] for x in batch]
        return p,pc
    def collect_fn3(batch):
        p=[x[0] for x in batch]
        c=[x[1] for x in batch]
        cc=[x[2] for x in batch]
        return p,c,cc
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True,collate_fn=collect_fn1)
    val_loader = DataLoader(test_dataset,
                            batch_size=args.batch,
                            shuffle=False,collate_fn=collect_fn1)
    p_loader = DataLoader(p_dataset, batch_size=args.batch, shuffle=True,collate_fn=collect_fn2)
    c_loader = DataLoader(c_dataset, batch_size=args.batch, shuffle=True,collate_fn=collect_fn3)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    poi_embed_model = PoiEmbeddings(num_pois, args.poi_embed_dim)

    # %% Model2: User embedding model, nn.embedding
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # %% Model3: Time Model
    time_embed_model = TimeEmbeddings(args.time_slot * 7, args.time_embed_dim)


    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    timeInterval_embed_model = TimeIntervalEmbeddings(args.transformer_hidden_dim,args.tu,args.tl)
    # %% Model5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

    word2vec = CAPE(num_pois, args.poi_embed_dim, num_cats, args.cat_embed_dim, args.device)

    # %% Model6: Sequence model
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim

    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_hidden_dim,
                                 args.transformer_nlayers,
                                 args.device,
                                 dropout=args.transformer_dropout)
    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(word2vec.parameters()) +
                                  list(timeInterval_embed_model.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training

    def input_traj_to_embeddings(users,\
                input_seqs_poi,input_seqs_cat,input_seqs_time_feature,label_timeInterval,input_seqs_timeMatrix,seqs_len,max_len):
        # Parse sample
        users=[[0]*(max_len-seqs_len[i])+[users[i]]*seqs_len[i] for i in range(len(users))]
        users=user_embed_model(torch.LongTensor(users).to(args.device))

        input_seqs_poi=poi_embed_model(torch.LongTensor(input_seqs_poi).to(args.device))
        input_seqs_cat=cat_embed_model(torch.LongTensor(input_seqs_cat).to(args.device))


        input_seqs_time_feature=time_embed_model(torch.LongTensor(input_seqs_time_feature).to(args.device))
        up_ct=torch.cat((embed_fuse_model1(users,input_seqs_poi),\
                         embed_fuse_model2(input_seqs_cat,input_seqs_time_feature)),dim=-1)

        label_timeInterval=timeInterval_embed_model.label_forward(torch.LongTensor(label_timeInterval).to(args.device),seqs_len,max_len)
        timeMatrixs=timeInterval_embed_model(torch.LongTensor(input_seqs_timeMatrix).to(args.device),seqs_len,max_len)

        return up_ct,timeMatrixs,label_timeInterval

    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    word2vec = word2vec.to(device=args.device)
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
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_cat_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf
    criterion = nn.LogSigmoid()
    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        word2vec.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_cat_loss_list = []

        # Loop batch
        for _, batch in enumerate(train_loader):

            users, input_seqs_poi, input_seqs_cat, \
                input_seqs_time_feature, \
                label_seqs_poi, label_seqs_cat, \
                label_seqs_timeInterval, input_seqs_timeMatrix, input_seqlens = batch

            seqs, timeMatrixs, label_timeIntervals = input_traj_to_embeddings(users,\
                input_seqs_poi,input_seqs_cat,input_seqs_time_feature,\
                            label_seqs_timeInterval,input_seqs_timeMatrix,input_seqlens,max_len)

            time_masks=torch.BoolTensor([[0]*x+[1]*(max_len-x) for x in input_seqlens]).to(args.device)
            y_poi=torch.LongTensor(label_seqs_poi).to(args.device)
            y_cat=torch.LongTensor(label_seqs_cat).to(args.device)

            y_pred_poi, y_pred_cat = seq_model(seqs,   label_timeIntervals,time_masks,timeMatrixs)
            print(y_poi.shape,y_cat.shape,y_pred_poi.shape,y_pred_cat.shape)
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            # Final loss
            loss = loss_poi + loss_cat
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
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
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, input_seqlens):
                label_pois = label_pois[max_len-seq_len:]  # shape: (seq_len, )
                pred_pois = pred_pois[max_len-seq_len:, :]  # shape: (seq_len, num_poi)
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
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

        for _, batch in enumerate(p_loader):
            target, context = batch
            input_target = torch.LongTensor(target).to(args.device)
            input_context = torch.LongTensor(context).to(args.device)
            input_target=poi_embed_model(input_target)
            positive, negative = word2vec(input_target, input_context, 16)
            # Optimizer Initialize
            loss = -(criterion(positive) + criterion(negative).mean()).sum()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

        for _, batch in enumerate(c_loader):
            poi, target, context = batch
            input_poi = torch.cuda.LongTensor(poi).to(args.device)
            input_target = torch.cuda.LongTensor(target).to(args.device)
            input_context = torch.cuda.LongTensor(context).to(args.device)
            input_poi=poi_embed_model(input_poi)
            input_target=cat_embed_model(input_target)
            # Optimizer Initialize
            positive, negative = word2vec.content(input_poi, input_target, input_context, 2)
            loss = -1 * args.ALPHA * (criterion(positive) + criterion(negative).mean()).sum()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)


        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        user_embed_model.eval()
        word2vec.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_cat_loss_list = []
        for _, batch in enumerate(val_loader):
            users, input_seqs_poi, input_seqs_cat, \
                input_seqs_time_feature, \
                label_seqs_poi, label_seqs_cat, \
                label_seqs_timeInterval, input_seqs_timeMatrix, input_seqlens = batch

            seqs, timeMatrixs, label_timeIntervals = input_traj_to_embeddings(users, \
                                                                              input_seqs_poi, input_seqs_cat,
                                                                              input_seqs_time_feature, \
                                                                              label_seqs_timeInterval,
                                                                              input_seqs_timeMatrix, input_seqlens,
                                                                              max_len)

            time_masks=torch.BoolTensor([[0]*x+[1]*(max_len-x) for x in input_seqlens]).to(args.device)
            y_poi = torch.LongTensor(label_seqs_poi).to(args.device)

            y_pred_poi, y_pred_cat = seq_model(seqs,   label_timeIntervals,time_masks,timeMatrixs)

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, input_seqlens):
                label_pois = label_pois[max_len-seq_len:]  # shape: (seq_len, )
                pred_pois = pred_pois[max_len-seq_len:, :]  # shape: (seq_len, num_poi)
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
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
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
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")


if __name__ == '__main__':
    args = parameter_parser()
    args.path='./dataset/dataset_TSMC2014_NYC.csv'
    args.drop_poi=5
    args.drop_user=5
    args.split_time = 60 * 60 * 6
    args.seqlen_thre=3
    args.context_size=1
    args.batch=32
    args.epochs=100
    args.device='cpu'
    args.seed=42
    args.poi_embed_dim=128
    args.cat_embed_dim=64
    args.user_embed_dim=128
    args.time_embed_dim=64
    args.time_slot=24
    args.transformer_hidden_dim=128
    args.transformer_nlayers=2
    args.transformer_dropout=0.1
    args.ALPHA=1
    args.tu=6*60*60
    args.tl=0

    train(args)
