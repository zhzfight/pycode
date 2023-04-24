import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from utils import sample_neighbors, split_list
import multiprocess as mp


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


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


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


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


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings and transform
    """

    def __init__(self, id2feat, device):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(MeanAggregator, self).__init__()
        self.id2feat = id2feat
        self.device = device

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
        for x, y in zip(row_indices, column_indices):
            mask[x, y] += 1

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        embed_matrix = self.id2feat(
            torch.LongTensor(list(unique_nodes_list)).to(self.device))  # （unique_count, feat_dim)
        to_feats = mask.mm(embed_matrix)  # n * embed_dim
        return to_feats  # n * embed_dim


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    id2feat -- function mapping LongTensor of node ids to FloatTensor of feature values.
    cuda -- whether to use GPU
    gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
    """

    def __init__(self, id2feat, adj_list, dis_list, restart_prob, num_walks, input_dim, output_dim, device, dropout,
                 workers,id):
        super(SageLayer, self).__init__()
        self.id2feat = id2feat
        self.dis_agg = MeanAggregator(self.id2feat, device)
        self.adj_agg = MeanAggregator(self.id2feat, device)
        self.device = device
        self.adj_list = adj_list
        self.dis_list = dis_list
        self.restart_prob = restart_prob
        self.num_walks = num_walks
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.workers = workers
        self.id=id
        self.W_self = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.W_adj = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.W_dis = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        # self.WC=nn.Linear(embed_dim,embed_dim)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W_self.weight.data.uniform_(-initrange, initrange)
        self.W_adj.weight.data.uniform_(-initrange, initrange)
        self.W_dis.weight.data.uniform_(-initrange, initrange)
        self.bias.data.zero_()

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        unique_nodes_list = list(set([int(node) for node in nodes]))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        if self.id==1:
            tasks = split_list(unique_nodes_list, self.workers)
            pool=mp.Pool(self.workers)
            feats=pool.map(self.help,tasks)
            feats=torch.cat(feats,dim=0)
        else:
            feats=self.help(unique_nodes_list)

        res = []
        for node in nodes:
            res.append(feats[unique_nodes[int(node)]])
        res = torch.stack(res, dim=0)
        return res
    def help(self, unique_nodes_list):
        adj_neighbors = sample_neighbors(self.adj_list, unique_nodes_list, self.restart_prob, self.num_walks, 'adj')
        dis_neighbors = sample_neighbors(self.dis_list, unique_nodes_list, self.restart_prob, self.num_walks, 'dis')
        self_feats = self.id2feat(torch.tensor(unique_nodes_list).to(self.device))
        adj_feats = self.adj_agg(adj_neighbors)
        dis_feats = self.dis_agg(dis_neighbors)
        adj_feats = self.W_adj(adj_feats)
        self_feats = self.W_self(self_feats)
        self_feats = F.dropout(self_feats, p=self.dropout, training=self.training)
        dis_feats = self.W_dis(dis_feats)
        feats = torch.cat((self_feats, adj_feats, dis_feats), dim=-1) + self.bias
        # feats=self.WC(feats)
        feats = self.leakyRelu(feats)
        feats = F.normalize(feats, p=2, dim=-1)
        return feats




class GraphSage(nn.Module):
    def __init__(self, X, num_node, embed_dim, adj, dis, device, restart_prob, num_walks, dropout, workers):
        super(GraphSage, self).__init__()
        self.id2node = X
        self.device = device
        '''
        self.layer1 = SageLayer(id2feat=lambda nodes: self.id2node[nodes], adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=X.shape[1],output_dim=embed_dim, device=device,
                                dropout=dropout,workers=workers,pool=self.pool)
        self.layer2 = SageLayer(id2feat=lambda nodes: self.layer1(nodes), adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks,input_dim=embed_dim,
                                output_dim=embed_dim, device=device, dropout=dropout,workers=workers,pool=self.pool)
        self.layer3 = SageLayer(id2feat=lambda nodes: self.layer2(nodes), adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks,input_dim=embed_dim,
                                output_dim=embed_dim, device=device, dropout=dropout,workers=workers,pool=self.pool)
        '''

        self.layer2 = SageLayer(id2feat=lambda nodes: self.id2node[nodes], adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=X.shape[1],
                                output_dim=embed_dim, device=device, dropout=dropout, workers=workers,id=2)
        self.layer1 = SageLayer(id2feat=lambda nodes: self.layer2(nodes), adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=embed_dim,
                                output_dim=embed_dim, device=device, dropout=dropout, workers=workers,id=1)

    def forward(self, nodes):
        feats = self.layer1(nodes)
        return feats


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_cat = self.decoder_cat(x)

        return out_poi, out_cat
