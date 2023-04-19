import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from torch.autograd import Variable
import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class PoiEmbeddings(nn.Module):
    def __init__(self,num_pois,embedding_dim):
        super(PoiEmbeddings,self).__init__()
        self.poi_embedding=nn.Embedding(num_embeddings=num_pois,embedding_dim=embedding_dim)
    def forward(self,poi_idx):
        embed=self.poi_embedding(poi_idx)
        return embed

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


class SpaAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings and transform
    """
    def __init__(self, id2feat,  device,id,feature_dim,embed_dim):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(SpaAggregator, self).__init__()

        self.id2feat = id2feat
        self.device=device
        self.id=id

        self.weight=nn.Parameter(torch.FloatTensor(feature_dim,embed_dim))
        self.bias=nn.Parameter(torch.FloatTensor(embed_dim))
        init.xavier_uniform_(self.weight)
        self.bias.data.zero_()

    def forward(self, nodes, adj_list, num_sample):
        """
        nodes --- list of nodes in a batch
        dis --- shape alike adj
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set  # a disordered non-repeatable list
        to_neighs=[list(adj.keys()) for adj in adj_list]
        # sample neighbors
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = []
            for i, to_neigh in enumerate(to_neighs):
                if len(to_neigh) > num_sample:
                    samp_neighs.append(_set(_sample(to_neigh, num_sample)))
                elif len(to_neigh)==0:
                    samp_neighs.append({int(nodes[i])})
                else:
                    samp_neighs.append(_set(to_neigh))
            # samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample
            #                else _set(to_neigh) for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        # ignore the unlinked nodes

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes))).to(self.device)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]

        #adj_weight=torch.tensor([adj_list[i][n] for i,samp_neigh in enumerate( samp_neighs) for n in samp_neigh]).to(self.device)
        #mask[row_indices, column_indices] = adj_weight  # can be replaced by distance
        mask[row_indices, column_indices] = 1
        # print(torch.sum(torch.isnan(mask)))
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        # spatial_transition = Variable(torch.FloatTensor(len(samp_neighs), len(unique_nodes)))
        # print(unique_nodes_list)
        # pdb.set_trace()
        if self.id==1:
            embed_matrix = self.id2feat[torch.LongTensor(unique_nodes_list).to(self.device)]  # （??, feat_dim)
        else:
            embed_matrix = self.id2feat(torch.LongTensor(unique_nodes_list).to(self.device))  # （??, feat_dim)
        embed_matrix = embed_matrix.mm(self.weight)+ self.bias
        to_feats = mask.mm(embed_matrix)  # (?, num_sample)
        # print(torch.sum(torch.isnan(embed_matrix)))
        return to_feats  # (?, feat_dim)


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    id2feat -- function mapping LongTensor of node ids to FloatTensor of feature values.
    cuda -- whether to use GPU
    gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
    """
    def __init__(self, id2feat, adj_list, context_sample_num, feature_dim,embed_dim, device, id):
        super(SageLayer, self).__init__()
        self.id=id
        self.id2feat = id2feat
        self.agg = SpaAggregator(self.id2feat,  device,id,feature_dim, embed_dim)
        self.num_sample = context_sample_num
        self.device=device
        self.adj_list = adj_list
        self.weight = nn.Parameter(
                torch.FloatTensor(feature_dim,embed_dim))
        self.wc=nn.Parameter(torch.FloatTensor(embed_dim,2*embed_dim))
        self.bias=nn.Parameter(torch.FloatTensor(embed_dim))
        self.bias.data.zero_()
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.wc)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """

        neigh_feats = self.agg(nodes, [self.adj_list[int(node)] for node in nodes], self.num_sample)
        if self.id==1:
            self_feats = self.id2feat[nodes]
        else:
            self_feats = self.id2feat(nodes)
        self_feats=self_feats.mm(self.weight)+self.bias
        combined = torch.cat((self_feats, neigh_feats), dim=1)  # (?, 2*feat_dim)
        # print(combined.shape)
        combined = F.relu(self.wc.mm(combined.t()))
        # pdb.set_trace()
        return combined

class GraphSage(nn.Module):
    def __init__(self, X,num_node, context_sample_num, embed_dim, adj, device):
        super(GraphSage, self).__init__()
        self.id2node = X
        self.device=device
        self.layer1 = SageLayer(self.id2node, adj, context_sample_num,self.id2node.shape[1], embed_dim, device, 1)
        self.layer12 = SageLayer(lambda nodes: self.layer1(nodes).t(), adj, context_sample_num,embed_dim, embed_dim, device, 2)


    def forward(self, nodes):
        neigh_embeds = self.layer12(torch.tensor(nodes).to(self.device)).t()  # (?, emb)
        return neigh_embeds

class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.decoder_context=nn.Linear(embed_size,num_poi)
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
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        out_context=self.decoder_context(x)



        return out_poi, out_time, out_cat,out_context
