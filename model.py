import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter


class PoiEmbeddings(nn.Module):
    def __init__(self,num_pois,embedding_dim):
        super(PoiEmbeddings,self).__init__()
        self.Poi_embedding=nn.Embedding(num_embeddings=num_pois+1,embedding_dim=embedding_dim,padding_idx=0)
    def forward(self,poi_ids):
        embed=self.Poi_embedding(poi_ids)
        return embed

class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users+1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats+1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), -1))
        x = self.leaky_relu(x)
        return x


class TimeIntervalEmbeddings(nn.Module):
    def __init__(self,dim,tu,tl):
        super(TimeIntervalEmbeddings,self).__init__()
        self.dim=dim
        self.emb_tu = torch.nn.Embedding(2, dim, padding_idx=0)
        self.emb_tl = torch.nn.Embedding(2, dim, padding_idx=0)
        self.tu=torch.LongTensor([tu])
        self.tl=torch.LongTensor([tl])
    def forward(self,delta_t,traj_len,max_len):
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, max_len-traj_len[i]:, max_len-traj_len[i]:] = 1
        etl, etu = self.emb_tl(mask), self.emb_tu(mask)
        vtl, vtu = (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.dim), \
            (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.dim)

        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        return time_interval
    def label_forward(self,delta_t,traj_len,max_len):
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, max_len - traj_len[i]:] = 1
        etl, etu = self.emb_tl(mask), self.emb_tu(mask)
        vtl, vtu = (delta_t - self.tl).unsqueeze(-1).expand( -1, -1, self.dim), \
            (self.tu - delta_t).unsqueeze(-1).expand( -1, -1, self.dim)

        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        return time_interval

class TimeEmbeddings(nn.Module):
    def __init__(self, time_slot,embedding_dim):
        super(TimeEmbeddings, self).__init__()

        self.time_embedding = nn.Embedding(
            num_embeddings=time_slot+1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

    def forward(self, time_features):
        embed = self.time_embedding(time_features)
        return embed

class Encoder(nn.Module):
    def __init__(self, input_dim,hidden_dim, dev, dropout=0.5):
        super(Encoder, self).__init__()
        self.Q_w = torch.nn.Linear(input_dim, hidden_dim)
        self.K_w = torch.nn.Linear(input_dim, hidden_dim)
        self.V_w = torch.nn.Linear(input_dim, hidden_dim)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_dim
        self.dropout_rate = dropout
        self.dev = dev


    def forward(self,batch_seq_embeds,time_mask,attn_mask,batch_seq_timeMatrix):
        Q, K, V = self.Q_w(batch_seq_embeds), self.K_w(batch_seq_embeds), self.V_w(batch_seq_embeds)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q.matmul(torch.transpose(K, 1, 2))
        attn_weights += batch_seq_timeMatrix.matmul(Q.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K.shape[-1] ** 0.5)


        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask.unsqueeze(-1).expand(-1, -1, attn_weights.shape[-1]), paddings, attn_weights)  # True:pick padding
        attn_weights=torch.where(time_mask.unsqueeze(1).expand(-1, attn_weights.shape[-1],-1), paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)  # enforcing causality

        attn_weights = self.softmax(attn_weights)  # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V)

        return outputs
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, input_dim, hidden_dim, nlayers, device, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.device=device
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-8)
        self.decoder_poi=torch.nn.Linear(hidden_dim, num_poi)
        self.decoder_cat=torch.nn.Linear(hidden_dim, num_cat)
        self.first_fwd_layer=torch.nn.Linear(input_dim,hidden_dim)

        for _ in range(nlayers):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = Encoder(hidden_dim,hidden_dim, self.device, dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_dim, dropout)
            self.forward_layers.append(new_fwd_layer)
    def forward(self,batch_seq_embeds,batch_seq_labels_timeInterval,time_mask,batch_seq_timeMatrix):
        tl = batch_seq_embeds.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        batch_seq_embeds=self.first_fwd_layer(batch_seq_embeds)
        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            print(batch_seq_embeds.shape)
            Q = self.attention_layernorms[i](batch_seq_embeds) # PyTorch mha requires time first fmt
            print(Q.shape)
            mha_outputs = self.attention_layers[i](Q, time_mask, attention_mask,
                                            batch_seq_timeMatrix)
            batch_seq_embeds = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            batch_seq_embeds = self.forward_layernorms[i](batch_seq_embeds)
            batch_seq_embeds = self.forward_layers[i](batch_seq_embeds)
            batch_seq_embeds *=  ~time_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(batch_seq_embeds)
        log_feats+=batch_seq_labels_timeInterval
        out_poi=self.decoder_poi(log_feats)
        out_cat=self.decoder_cat(log_feats)
        return out_poi,out_cat


class CAPE(nn.Module):
    def __init__(self,  poi_num,poi_embed_dim,cat_num,cat_embed_dim,dev ):
        super(CAPE, self).__init__()
        # Embedding Matrix
        self.poi_num=poi_num
        self.cat_num=cat_num
        self.dev=dev
        self.poi_embedding_out = nn.Embedding(poi_num, poi_embed_dim,padding_idx=0)
        self.cat_embedding_out = nn.Embedding(cat_num, cat_embed_dim,padding_idx=0)
        # FF Layer
        self.ff = nn.Linear(poi_embed_dim+cat_embed_dim, cat_embed_dim)

    def forward(self, embedded_poi_in, context, num_sampled=None):
        # Embedding Lookup

        embedded_poi_out = self.poi_embedding_out(context)

        # =============================
        # Positive Loss
        # =============================
        target_loss = (embedded_poi_in * embedded_poi_out).sum(1).squeeze()

        # =============================
        # Negative Loss
        # =============================

        # Negative Sampling
        batch_size = embedded_poi_in.shape[0]
        negative_samples = torch.FloatTensor(batch_size, num_sampled).to(self.dev).uniform_(1, self.poi_num).long()
        embedded_samples = self.poi_embedding_out(negative_samples).neg()

        negative_loss = torch.bmm(embedded_samples, embedded_poi_in.unsqueeze(2))

        return target_loss, negative_loss
    def cat(self,embedded_poi_in,embedded_cat_in,cat_context,num_sampled=None):
        # Embedding Lookup
        embedded_cat_context = self.cat_embedding_out(cat_context)
        # Concat
        mixed_target = torch.cat([embedded_poi_in, embedded_cat_in], 1)
        mixed_target = self.ff(mixed_target)
        # =============================
        # Positive Loss
        # =============================
        target_loss = (mixed_target * embedded_cat_context).sum(1).squeeze()
        # =============================
        # Negative Loss
        # =============================
        # Negative Sampling
        batch_size = embedded_poi_in.shape[0]
        negative_samples = torch.FloatTensor(batch_size, num_sampled).to(self.dev).uniform_(1, self.cat_num).long()
        embedded_samples = self.word_embedding_out(negative_samples).neg()
        negative_loss = torch.bmm(embedded_samples, mixed_target.unsqueeze(2))
        return target_loss, negative_loss

