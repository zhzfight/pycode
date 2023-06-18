import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import UserEmbeddings, CategoryEmbeddings, TimeIntervalAwareTransformer, PoiEmbeddings, TimeEmbeddings
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss


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

    pois_in_train = set()

    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, df):
            self.users = []
            self.input_seqs = []
            self.label_seqs = []

            input_end_idx = -3

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
                pois_in_train.update(poi_ids)
                input_start_idx = max(len(poi_ids)-3 - args.max_seq_len, 0)
                label_start_idx = input_start_idx + 1
                self.users.append(user)
                self.input_seqs.append(list(
                    zip(poi_ids[input_start_idx:input_end_idx], time_bins[input_start_idx:input_end_idx],
                        h[input_start_idx:input_end_idx], w[input_start_idx:input_end_idx])))
                self.label_seqs.append(list(
                    zip(poi_ids[label_start_idx:label_end_idx], time_bins[label_start_idx:label_end_idx],
                        h[label_start_idx:label_end_idx], w[label_start_idx:label_end_idx])))

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (self.users[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.users = []
            self.input_seqs = []
            self.label_seqs = []
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
                self.input_seqs.append(list(
                    zip(poi_ids[input_start_idx:input_end_idx], time_bins[input_start_idx:input_end_idx],
                        h[input_start_idx:input_end_idx], w[input_start_idx:input_end_idx])))
                self.label_seqs.append(list(
                    zip(poi_ids[label_start_idx:label_end_idx], time_bins[label_start_idx:label_end_idx],
                        h[label_start_idx:label_end_idx], w[label_start_idx:label_end_idx])))

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (self.users[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetTest(Dataset):
        def __init__(self, df):
            self.users = []
            self.input_seqs = []
            self.label_seqs = []
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
                self.input_seqs.append(list(
                    zip(poi_ids[input_start_idx:input_end_idx], time_bins[input_start_idx:input_end_idx],
                        h[input_start_idx:input_end_idx], w[input_start_idx:input_end_idx])))
                self.label_seqs.append(list(
                    zip(poi_ids[label_start_idx:], time_bins[label_start_idx:],
                        h[label_start_idx:], w[label_start_idx:])))

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.users)
            return len(self.users)

        def __getitem__(self, index):
            return (self.users[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(df)
    val_dataset = TrajectoryDatasetVal(df)
    test_dataset = TrajectoryDatasetTest(df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)
    test_loader = DataLoader(val_dataset,
                             batch_size=args.batch,
                             shuffle=False, drop_last=False,
                             pin_memory=True, num_workers=args.workers,
                             collate_fn=lambda x: x)

    # %% ====================== Build Models ======================

    poi_embed_model = PoiEmbeddings(poi_num, args.poi_embed_dim)

    user_embed_model = UserEmbeddings(user_num, args.user_embed_dim)

    # %% Model3: Time Model
    time_embed_model = TimeEmbeddings(args.time_embed_dim)

    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(cat_num, args.cat_embed_dim)

    # %% Model6: Sequence model
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    seq_model = TimeIntervalAwareTransformer(num_poi=poi_num,
                                             num_cat=cat_num,
                                             nhid=args.seq_input_embed,
                                             batch_size=args.batch,
                                             device=args.device,
                                             dropout=args.dropout)

    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def input_traj_to_embeddings(sample):
        user = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_cat = [poi2cat[each] for each in input_seq]
        input_seq_time = [each[1] for each in sample[1]]

        poiid_embedded = poi_embed_model(torch.LongTensor(input_seq).to(args.device))
        catid_embedded = cat_embed_model(torch.LongTensor(input_seq_cat).to(args.device))
        timebin_embeded = time_embed_model(torch.LongTensor(input_seq_time).to(args.device))
        user_embedded = user_embed_model(torch.LongTensor([user]).to(args.device))
        user_embedded = user_embedded.repeat(poiid_embedded.shape[0],1)
        embedded = torch.cat((poiid_embedded, catid_embedded, timebin_embeded, user_embedded), dim=-1)

        return embedded

    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
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

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
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

        # Loop batch
        for b_idx, batch in enumerate(train_loader):

            # For padding
            batch_input_seqs_h = []
            batch_input_seqs_w = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_label_seqs = []
            batch_label_seqs_h=[]
            batch_label_seqs_w=[]

            # Convert input seq to embeddings
            for sample in batch:
                input_seq_h = [each[2] for each in sample[1]]
                batch_input_seqs_h.append(input_seq_h)
                input_seq_w=[each[3] for each in sample[1]]
                batch_input_seqs_w.append(input_seq_w)
                label_seq=[each[0] for each in sample[2]]
                batch_label_seqs.append(torch.LongTensor(label_seq).to(args.device))
                label_seq_h = [each[2] for each in sample[2]]
                batch_label_seqs_h.append(label_seq_h)
                label_seq_w = [each[3] for each in sample[2]]
                batch_label_seqs_w.append(label_seq_w)
                input_seq_embed = input_traj_to_embeddings(sample)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(label_seq))


            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_label_seqs, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_pred_poi = seq_model(x, batch_seq_lens, batch_input_seqs_h,batch_input_seqs_w,batch_label_seqs_h,batch_label_seqs_w)

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
            if (b_idx % (100)) == 0:
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
        poi_embed_model.eval()

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

        for vb_idx, batch in enumerate(val_loader):

            # For padding
            batch_input_seqs_h = []
            batch_input_seqs_w = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_label_seqs = []
            batch_label_seqs_h = []
            batch_label_seqs_w = []

            # Convert input seq to embeddings
            for sample in batch:
                input_seq_h = [each[2] for each in sample[1]]
                batch_input_seqs_h.append(input_seq_h)
                input_seq_w = [each[3] for each in sample[1]]
                batch_input_seqs_w.append(input_seq_w)
                label_seq = [each[0] for each in sample[2]]
                batch_label_seqs.append(torch.LongTensor(label_seq).to(args.device))
                label_seq_h = [each[2] for each in sample[2]]
                batch_label_seqs_h.append(label_seq_h)
                label_seq_w = [each[3] for each in sample[2]]
                batch_label_seqs_w.append(label_seq_w)
                input_seq_embed = input_traj_to_embeddings(sample)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(label_seq))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_label_seqs, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_pred_poi = seq_model(x, batch_seq_lens, batch_input_seqs_h,batch_input_seqs_w,batch_label_seqs_h,batch_label_seqs_w)

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
            if (vb_idx % (200)) == 0:
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


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    train(args)
