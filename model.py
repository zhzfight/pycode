import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from math import pi, log

from einops import rearrange, repeat
import queue
from utils import sample_neighbors


class PoiEmbeddings(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(PoiEmbeddings, self).__init__()
        self.poi_embedding = nn.Embedding(num_embeddings=num_pois, embedding_dim=embedding_dim)

    def forward(self, poi_idx):
        return self.poi_embedding(poi_idx)


class TimeEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbeddings, self).__init__()
        self.time_embedding = nn.Embedding(num_embeddings=24 * 7, embedding_dim=embedding_dim)

    def forward(self, time_idx):
        return self.time_embedding(time_idx)


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class TimeIntervalAwareTransformer(nn.Module):
    def __init__(self, num_poi, num_cat, nhid, batch_size, device, dropout, user_dim):
        super(TimeIntervalAwareTransformer, self).__init__()

        self.device = device
        self.nhid = nhid
        self.batch_size = batch_size
        # self.encoder = nn.Embedding(num_poi, embed_size)

        self.decoder_poi = nn.Linear(nhid, num_poi)

        self.day_embedding = nn.Embedding(8, nhid, padding_idx=0)
        self.hour_embedding = nn.Embedding(25, nhid, padding_idx=0)

        self.label_day_embedding = nn.Embedding(8, nhid, padding_idx=0)
        self.label_hour_embedding = nn.Embedding(25, nhid, padding_idx=0)

        self.W1_Q = nn.Linear(nhid, nhid)
        self.W1_K = nn.Linear(nhid, nhid)
        self.W1_V = nn.Linear(nhid, nhid)
        self.norm11 = nn.LayerNorm(nhid)
        self.feedforward1 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid)
        )
        self.norm12 = nn.LayerNorm(nhid)

        self.W2_Q = nn.Linear(nhid, nhid)
        self.W2_K = nn.Linear(nhid, nhid)
        self.W2_V = nn.Linear(nhid, nhid)
        self.norm21 = nn.LayerNorm(nhid)
        self.feedforward2 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid)
        )
        self.norm22 = nn.LayerNorm(nhid)

        self.init_weights()
        self.rotary_emb_attn = RotaryEmbedding(dim=nhid,device=device)
        self.rotary_emb_decode = RotaryEmbedding(dim=nhid,device=device)
        self.u_proj = nn.Linear(user_dim, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, batch_seq_lens, batch_input_h_matrices, batch_input_w_matrices, batch_label_h_matrices,
                batch_label_w_matrices, batch_user_embedding,batch_seq_idxes):
        hourInterval_embedding = self.hour_embedding(batch_input_h_matrices)
        dayInterval_embedding = self.day_embedding(batch_input_w_matrices)
        label_hourInterval_embedding = self.hour_embedding(batch_label_h_matrices)
        label_dayInterval_embedding = self.day_embedding(batch_label_w_matrices)
        # mask attn
        attn_mask = ~torch.tril(torch.ones((src.shape[1], src.shape[1]), dtype=torch.bool, device=self.device))
        time_mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool, device=self.device)
        for i in range(src.shape[0]):
            time_mask[i, batch_seq_lens[i]:] = True

        attn_mask = attn_mask.unsqueeze(0).expand(src.shape[0], -1, -1)
        time_mask = time_mask.unsqueeze(-1).expand(-1, -1, src.shape[1])

        src = self.pos_encoder(src)
        # src=self.rotary_emb_attn.rotate_queries_or_keys(src)
        Q = self.W1_Q(src)
        K = self.W1_K(src)
        V = self.W1_V(src)

        attn_weight = Q.matmul(torch.transpose(K, 1, 2))
        attn_weight += hourInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weight += dayInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)

        attn_weight = attn_weight / math.sqrt(self.nhid)

        paddings = torch.ones(attn_weight.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.device)

        attn_weight = torch.where(time_mask, paddings, attn_weight)
        attn_weight = torch.where(attn_mask, paddings, attn_weight)

        attn_weight = F.softmax(attn_weight, dim=-1)
        x = attn_weight.matmul(V)  # B,L,D
        x += torch.matmul(attn_weight.unsqueeze(2), hourInterval_embedding).squeeze(2)
        x += torch.matmul(attn_weight.unsqueeze(2), dayInterval_embedding).squeeze(2)

        x = self.norm11(x + src)
        ffn_output = self.feedforward1(x)
        ffn_output = self.norm12(x + ffn_output)

        src = ffn_output

        Q = self.W2_Q(src)
        K = self.W2_K(src)
        V = self.W2_V(src)

        attn_weight = Q.matmul(torch.transpose(K, 1, 2))
        attn_weight += hourInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weight += dayInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weight = attn_weight / math.sqrt(self.nhid)
        paddings = torch.ones(attn_weight.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.device)

        attn_weight = torch.where(time_mask, paddings, attn_weight)
        attn_weight = torch.where(attn_mask, paddings, attn_weight)

        attn_weight = F.softmax(attn_weight, dim=-1)
        x = attn_weight.matmul(V)  # B,L,D
        x += torch.matmul(attn_weight.unsqueeze(2), hourInterval_embedding).squeeze(2)
        x += torch.matmul(attn_weight.unsqueeze(2), dayInterval_embedding).squeeze(2)

        x = self.norm21(x + src)
        ffn_output = self.feedforward2(x)
        ffn_output = self.norm22(x + ffn_output)

        # attn_mask=attn_mask.unsqueeze(-1).expand(-1,-1,-1,ffn_output.shape[-1])
        ffn_output = ffn_output.unsqueeze(2).repeat(1, 1, ffn_output.shape[1], 1).transpose(2, 1)
        ffn_output = torch.add(ffn_output, label_hourInterval_embedding)
        ffn_output = torch.add(ffn_output, label_dayInterval_embedding)
        '''
        decoder_output_poi = self.decoder_poi(ffn_output)
        pooled_poi=torch.zeros(decoder_output_poi.shape[0],decoder_output_poi.shape[1],decoder_output_poi.shape[3]).to(self.device)
        for i in range(decoder_output_poi.shape[1]):
            pooled_poi[:,i]=torch.mean(decoder_output_poi[:,i,:i+1],dim=1)

        '''
        batch_user_embedding = self.u_proj(batch_user_embedding)
        batch_user_embedding = batch_user_embedding.unsqueeze(1).repeat(1, src.shape[1], 1)
        attention_weights = torch.zeros(src.shape[0], src.shape[1], src.shape[1]).to(self.device)
        for i in range(src.shape[1]):
            attention_weights[:, i, :i + 1] = F.softmax(
                torch.sum(ffn_output[:, i, :i + 1] * batch_user_embedding[:, :i + 1], dim=-1), dim=-1)

        # 根据注意力权重对偏好矩阵进行聚合
        aggregated_preference = torch.sum(ffn_output * attention_weights.unsqueeze(-1), dim=2)
        pooled_poi = self.decoder_poi(aggregated_preference)

        return pooled_poi


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            device,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
            use_xpos=False,
            xpos_scale_base=512,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.device=device

    def rotate_queries_or_keys(self, t,idx):
        freqs = self.forward(idx)
        return self.apply_rotary_emb(freqs, t)

    def forward(self, idx):
        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', idx, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        return freqs

    def rotate_half(self,x):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    def apply_rotary_emb(self,freqs, t,  scale=1.):
        rot_dim = freqs.shape[-1]
        t = (t * freqs.cos() * scale) + (self.rotate_half(t) * freqs.sin() * scale)
        return t


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings and transform
    """

    def __init__(self, id2feat, device, dim):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(MeanAggregator, self).__init__()
        self.id2feat = id2feat
        self.device = device
        self.W = nn.Linear(dim, dim)

    def forward(self, to_neighs):
        """
        nodes --- list of nodes in a batch
        dis --- shape alike adj
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        tmp = [n for x in to_neighs for n in x]
        unique_nodes_list = set(tmp)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(to_neighs), len(unique_nodes_list)).to(self.device)
        column_indices = [unique_nodes[n] for n in tmp]
        row_indices = [i for i in range(len(to_neighs)) for j in range(len(to_neighs[i]))]

        mask[row_indices, column_indices] += 1

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        embed_matrix = self.id2feat(
            torch.LongTensor(list(unique_nodes_list)).to(self.device))  # （unique_count, feat_dim)
        embed_matrix = self.W(embed_matrix)
        to_feats = mask.mm(embed_matrix)  # n * embed_dim
        return to_feats  # n * embed_dim


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    id2feat -- function mapping LongTensor of node ids to FloatTensor of feature values.
    cuda -- whether to use GPU
    gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
    """

    def __init__(self, id2feat, restart_prob, num_walks, input_dim, output_dim, device, dropout,
                 id, adj_dicts, dis_dicts):
        super(SageLayer, self).__init__()
        self.id2feat = id2feat
        self.dis_agg = MeanAggregator(self.id2feat, device, input_dim)
        self.adj_agg = MeanAggregator(self.id2feat, device, input_dim)
        self.device = device
        self.adj_list = None
        self.dis_list = None
        self.restart_prob = restart_prob
        self.num_walks = num_walks
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.adj_dicts = adj_dicts
        self.dis_dicts = dis_dicts
        self.id = id
        self.W_self = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.W_adj = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.W_dis = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.WC = nn.Linear(output_dim, output_dim)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W_self.weight.data.uniform_(-initrange, initrange)
        self.W_adj.weight.data.uniform_(-initrange, initrange)
        self.W_dis.weight.data.uniform_(-initrange, initrange)
        self.WC.weight.data.uniform_(-initrange, initrange)
        self.bias.data.zero_()

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """

        unique_nodes_list = list(set([int(node) for node in nodes]))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        adj_neighbors = [[] for _ in unique_nodes_list]
        dis_neighbors = [[] for _ in unique_nodes_list]
        missing_adj_idx = []
        missing_dis_idx = []
        for idx, node in enumerate(unique_nodes_list):
            try:
                random_walk = self.adj_dicts[node].get(block=False)
                adj_neighbors[idx] = random_walk
            except queue.Empty:
                missing_adj_idx.append(idx)
            try:
                random_walk = self.dis_dicts[node].get(block=False)
                dis_neighbors[idx] = random_walk
            except queue.Empty:
                missing_dis_idx.append(idx)
        if len(missing_adj_idx) != 0:
            missing_adj_neighbors = sample_neighbors(self.adj_list, [unique_nodes_list[i] for i in missing_adj_idx],
                                                     self.restart_prob, self.num_walks, 'adj')
            for idx, missing_adj_neighbor in zip(missing_adj_idx, missing_adj_neighbors):
                adj_neighbors[idx] = missing_adj_neighbor
        if len(missing_dis_idx) != 0:
            missing_dis_neighbors = sample_neighbors(self.dis_list, [unique_nodes_list[i] for i in missing_dis_idx],
                                                     self.restart_prob, self.num_walks, 'dis')
            for idx, missing_dis_neighbor in zip(missing_dis_idx, missing_dis_neighbors):
                dis_neighbors[idx] = missing_dis_neighbor

        self_feats = self.id2feat(torch.tensor(unique_nodes_list).to(self.device))
        adj_feats = self.adj_agg(adj_neighbors)
        dis_feats = self.dis_agg(dis_neighbors)

        adj_feats = self.W_adj(adj_feats)
        self_feats = self.W_self(self_feats)
        self_feats = F.dropout(self_feats, p=self.dropout, training=self.training)
        dis_feats = self.W_dis(dis_feats)
        feats = torch.cat((self_feats, adj_feats, dis_feats), dim=-1)
        feats = self.WC(feats) + self.bias
        feats = self.leakyRelu(feats)
        feats = F.normalize(feats, p=2, dim=-1)

        nodes_idx = [unique_nodes[int(node)] for node in nodes]
        res = feats[nodes_idx]

        return res

    def set_adj(self, adj, dis):
        self.adj_list = adj
        self.dis_list = dis


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, embed_dim, device, restart_prob, num_walks, dropout, adj_dicts, dis_dicts):
        super(GraphSAGE, self).__init__()
        self.id2node = None
        self.device = device

        self.layer2 = SageLayer(id2feat=lambda nodes: self.id2node[nodes],
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=input_dim,
                                output_dim=embed_dim, device=device, dropout=dropout, id=2, adj_dicts=adj_dicts,
                                dis_dicts=dis_dicts)
        self.layer1 = SageLayer(id2feat=lambda nodes: self.layer2(nodes),
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=embed_dim,
                                output_dim=embed_dim, device=device, dropout=dropout, id=1, adj_dicts=adj_dicts,
                                dis_dicts=dis_dicts)

    def forward(self, nodes):
        feats = self.layer1(nodes)
        return feats

    def setup(self, X, adj, dis):
        self.id2node = X
        self.layer1.set_adj(adj, dis)
        self.layer2.set_adj(adj, dis)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, device, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.device = device
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, batch_seq_lens, batch_input_h_matrices, batch_input_w_matrices, batch_label_h_matrices,
                batch_label_w_matrices, batch_user_embedding,batch_seq_idxes):
        src_mask = self.generate_square_subsequent_mask(src.shape[1]).to(self.device)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        return out_poi


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, nhid, dropout_rate, device):
        super(Attention, self).__init__()
        self.Q_w = torch.nn.Linear(nhid, nhid)
        self.K_w = torch.nn.Linear(nhid, nhid)
        self.V_w = torch.nn.Linear(nhid, nhid)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_size = nhid
        self.dropout_rate = dropout_rate
        self.device = device
        self.rotary_embed=RotaryEmbedding(dim=nhid,device=device)

    def forward(self, seqs, time_mask, attn_mask, d_time_matrix_K, w_time_matrix_K, d_time_matrix_V,
                w_time_matrix_V,batch_seq_idxes):
        Q, K, V = self.Q_w(seqs), self.K_w(seqs), self.V_w(seqs)
        Q_=self.rotary_embed.rotate_queries_or_keys(Q,batch_seq_idxes)
        K_=self.rotary_embed.rotate_queries_or_keys(K,batch_seq_idxes)
        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += d_time_matrix_K.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weights += w_time_matrix_K.matmul(Q.unsqueeze(-1)).squeeze(-1)
        # seq length adaptive scaling
        attn_weights = attn_weights / (K.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.device)
        attn_weights = torch.where(time_mask, paddings, attn_weights)  # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights)  # enforcing causality

        attn_weights = self.softmax(attn_weights)  # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V)
        outputs += attn_weights.unsqueeze(2).matmul(d_time_matrix_V).reshape(outputs.shape).squeeze(2)
        outputs += attn_weights.unsqueeze(2).matmul(w_time_matrix_V).reshape(outputs.shape).squeeze(2)

        return outputs


class IntervalAwareTransformer(nn.Module):
    def __init__(self, num_poi, num_cat, nhid, batch_size, device, dropout, user_dim, max_len, nlayer):
        super(IntervalAwareTransformer, self).__init__()
        self.num_poi = num_poi
        self.device = device
        self.decoder_poi=nn.Linear(nhid,num_poi)


        self.d_time_matrix_K_emb = torch.nn.Embedding(25, nhid)
        self.d_time_matrix_V_emb = torch.nn.Embedding(25, nhid)
        self.w_time_matrix_K_emb = torch.nn.Embedding(8, nhid)
        self.w_time_matrix_V_emb = torch.nn.Embedding(8, nhid)


        self.d_time_matrix_K_dropout = torch.nn.Dropout(p=dropout)
        self.d_time_matrix_V_dropout = torch.nn.Dropout(p=dropout)
        self.w_time_matrix_K_dropout = torch.nn.Dropout(p=dropout)
        self.w_time_matrix_V_dropout = torch.nn.Dropout(p=dropout)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(nhid, eps=1e-8)

        for _ in range(nlayer):
            new_attn_layernorm = torch.nn.LayerNorm(nhid, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = Attention(nhid,
                                       dropout,
                                       device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(nhid, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(nhid, dropout)
            self.forward_layers.append(new_fwd_layer)


    def forward(self,seqs, batch_seq_lens, batch_input_h_matrices, batch_input_w_matrices, batch_label_h_matrices,
                batch_label_w_matrices, batch_user_embedding,batch_seq_idxes):


        d_time_matrix_K = self.d_time_matrix_K_emb(batch_input_h_matrices)
        d_time_matrix_V = self.d_time_matrix_V_emb(batch_input_h_matrices)
        d_time_matrix_K = self.d_time_matrix_K_dropout(d_time_matrix_K)
        d_time_matrix_V = self.d_time_matrix_V_dropout(d_time_matrix_V)

        w_time_matrix_K = self.w_time_matrix_K_emb(batch_input_w_matrices)
        w_time_matrix_V = self.w_time_matrix_V_emb(batch_input_w_matrices)
        w_time_matrix_K = self.w_time_matrix_K_dropout(w_time_matrix_K)
        w_time_matrix_V = self.w_time_matrix_V_dropout(w_time_matrix_V)


        # mask attn
        attention_mask = ~torch.tril(torch.ones((seqs.shape[1], seqs.shape[1]), dtype=torch.bool, device=self.device))
        timeline_mask = torch.zeros((seqs.shape[0], seqs.shape[1]), dtype=torch.bool, device=self.device)
        for i in range(seqs.shape[0]):
            timeline_mask[i, batch_seq_lens[i]:] = True

        for i in range(len(self.attention_layers)):

            mha_outputs = self.attention_layers[i](seqs,
                                                   timeline_mask, attention_mask,
                                                   d_time_matrix_K, w_time_matrix_K,d_time_matrix_V,w_time_matrix_V,batch_seq_idxes)
            seqs = seqs + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        score=self.decoder_poi(log_feats)

        return score
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