import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


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





class GRUModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size,nhid,batch_size, node_attn_in_features,node_attn_nhid,device,dropout,tu):
        super(GRUModel, self).__init__()
        from torch.nn import GRU
        self.time_embed_size=50
        self.dropout=nn.Dropout(dropout)
        self.grucell = nn.GRUCell(embed_size, nhid)
        self.h0 = self.h0 = nn.Parameter(torch.randn(1, nhid)).to(device)
        self.node_attn_model=NodeAttnMap(in_features=node_attn_in_features, nhid=node_attn_nhid, use_mask=False)
        self.device=device
        self.nhid=nhid
        self.batch_size=batch_size
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(nhid, num_poi)
        self.decoder_time = nn.Linear(nhid, 1)
        self.decoder_cat = nn.Linear(nhid, num_cat)
        self.init_weights()
        self.emb_tu = nn.Embedding(2, self.time_embed_size, padding_idx=0)
        self.emb_tl = nn.Embedding(2, self.time_embed_size, padding_idx=0)
        self.tu=tu



    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src,batch_seq_lens,batch_input_seqs, X,A):
        attns = torch.full((src.shape[0], src.shape[1], src.shape[1]), -1e9).to(self.device)
        attn_map = self.node_attn_model(X, A)
        # attn caculate
        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]
            for j in range(1,len(traj_i_input)):
                for k in range( j):
                    attns[i, j,k] = attn_map[traj_i_input[k], traj_i_input[j]]

        v0,indices=torch.max(attns,dim=-1)

        v1=torch.zeros((src.shape[0],src.shape[1])).to(self.device)
        for i in range(1, src.shape[1]):
            v1[:,i]=attns[:,i,i-1]


        v= torch.stack([v0, v1], dim=-1)
        v = torch.nn.functional.softmax(v, dim=-1)


        hid = self.h0.repeat(src.shape[0], 1).to(self.device)
        x = torch.zeros((src.shape[0],src.shape[1],self.nhid)).to(self.device)
        hid = self.grucell(src[:,0,:],hid)
        x[:,0,:]=hid
        for i in range(1,src.shape[1]):
            hid1 = self.grucell(src[:, i, :], hid)
            hid2 = self.grucell(src[:,i,:],x[np.arange(x.shape[0]), indices[:, i]])


            alpha1=v[:,i,0].unsqueeze(-1).repeat(1,self.nhid)
            alpha2=v[:,i,1].unsqueeze(-1).repeat(1,self.nhid)
            hid = torch.mul(alpha1,hid1)+torch.mul(alpha2,hid2)
            hid=self.dropout(hid)
            x[:,i,:]=hid


        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat
