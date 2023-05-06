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
        print(len(channels))
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
    def __init__(self, num_poi, num_cat, nhid,batch_size, device,dropout):
        super(GRUModel, self).__init__()


        self.device=device
        self.nhid=nhid
        self.batch_size=batch_size
        # self.encoder = nn.Embedding(num_poi, embed_size)

        self.decoder_poi = nn.Linear(nhid, num_poi)
        self.decoder_cat = nn.Linear(nhid, num_cat)
        self.tu=24*3600
        self.time_bin=3600*2
        assert (self.tu)%self.time_bin==0
        self.day_embedding=nn.Embedding(8,nhid,padding_idx=0)
        self.hour_embedding=nn.Embedding(int((self.tu)/self.time_bin)+2,nhid,padding_idx=0)

        self.W1_Q=nn.Linear(nhid,nhid)
        self.W1_K=nn.Linear(nhid,nhid)
        self.W1_V=nn.Linear(nhid,nhid)
        self.norm11=nn.LayerNorm(nhid)
        self.feedforward1 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid)
        )
        self.norm12 = nn.LayerNorm(nhid)

        self.W2_Q=nn.Linear(nhid,nhid)
        self.W2_K=nn.Linear(nhid,nhid)
        self.W2_V = nn.Linear(nhid, nhid)
        self.norm21=nn.LayerNorm(nhid)
        self.feedforward2 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid)
        )
        self.norm22 = nn.LayerNorm(nhid)



        self.init_weights()




    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src,batch_seq_lens,batch_input_seqs_ts,batch_label_seqs_ts):
        hourInterval=torch.zeros((src.shape[0],src.shape[1],src.shape[1]),dtype=torch.long).to(self.device)
        dayInterval=torch.zeros((src.shape[0],src.shape[1],src.shape[1]),dtype=torch.long).to(self.device)

        label_hourInterval=torch.zeros((src.shape[0],src.shape[1],src.shape[1]),dtype=torch.long).to(self.device)
        label_dayInterval=torch.zeros((src.shape[0],src.shape[1],src.shape[1]),dtype=torch.long).to(self.device)
        for i in range(src.shape[0]):
            for j in range(batch_seq_lens[i]):
                for k in range(j+1):
                    if i==j:
                        hourInterval[i][j][k]=1
                    else:
                        hourInterval[i][j][k]=int(((batch_input_seqs_ts[i][j]-batch_input_seqs_ts[i][k])%(self.tu))/self.time_bin)+2
                    dayInterval[i][j][k]=int((batch_input_seqs_ts[i][j]-batch_input_seqs_ts[i][k])/(self.tu))+1
                    if dayInterval[i][j][k]>6:
                        dayInterval[i][j][k]=7
                    label_hourInterval[i][j][k] = int(
                        ((batch_label_seqs_ts[i][j] - batch_input_seqs_ts[i][k]) % (self.tu)) / self.time_bin) + 2
                    label_dayInterval[i][j][k] = int(
                        (batch_label_seqs_ts[i][j] - batch_input_seqs_ts[i][k]) / (self.tu)) + 1
                    if label_dayInterval[i][j][k] > 6:
                        label_dayInterval[i][j][k] = 7




        hourInterval_embedding=self.hour_embedding(hourInterval)
        dayInterval_embedding=self.day_embedding(dayInterval)

        label_hourInterval_embedding=self.hour_embedding(label_hourInterval)
        label_dayInterval_embedding=self.day_embedding(label_dayInterval)

        # mask attn
        attn_mask = ~torch.tril(torch.ones((src.shape[1], src.shape[1]), dtype=torch.bool, device=self.device))
        time_mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool, device=self.device)
        for i in range(src.shape[0]):
            time_mask[i, batch_seq_lens[i]:] = True

        attn_mask = attn_mask.unsqueeze(0).expand(src.shape[0], -1, -1)
        time_mask = time_mask.unsqueeze(-1).expand(-1, -1, src.shape[1])

        Q=self.W1_Q(src)
        K=self.W1_K(src)
        V=self.W1_V(src)

        attn_weight=Q.matmul(torch.transpose(K,1,2))
        attn_weight+=hourInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weight+=dayInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)

        attn_weight=attn_weight/math.sqrt(self.nhid)

        paddings = torch.ones(attn_weight.shape) * (-2 ** 32 + 1)
        paddings=paddings.to(self.device)

        attn_weight=torch.where(time_mask,paddings,attn_weight)
        attn_weight=torch.where(attn_mask,paddings,attn_weight)


        attn_weight=F.softmax(attn_weight,dim=-1)
        x=attn_weight.matmul(V) #B,L,D
        x+=torch.matmul(attn_weight.unsqueeze(2),hourInterval_embedding).squeeze(2)
        x+=torch.matmul(attn_weight.unsqueeze(2),dayInterval_embedding).squeeze(2)


        x=self.norm11(x+src)
        ffn_output=self.feedforward1(x)
        ffn_output=self.norm12(x+ffn_output)
        '''
        src=ffn_output

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

        '''
        #attn_mask=attn_mask.unsqueeze(-1).expand(-1,-1,-1,ffn_output.shape[-1])
        ffn_output=ffn_output.unsqueeze(2).repeat(1,1,ffn_output.shape[1],1).transpose(2,1)
        ffn_output=torch.add(ffn_output,label_hourInterval_embedding)
        ffn_output=torch.add(ffn_output,label_dayInterval_embedding)
        '''
        paddings = torch.ones(ffn_output.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.device)
        ffn_output = torch.where(attn_mask, paddings, ffn_output)
        '''
        decoder_output_poi = self.decoder_poi(ffn_output)
        decoder_output_cat = self.decoder_cat(ffn_output)
        pooled_poi=torch.zeros(decoder_output_poi.shape[0],decoder_output_poi.shape[1],decoder_output_poi.shape[3]).to(self.device)
        pooled_cat=torch.zeros(decoder_output_cat.shape[0],decoder_output_cat.shape[1],decoder_output_cat.shape[3]).to(self.device)
        for i in range(decoder_output_poi.shape[1]):
            pooled_poi[:,i]=torch.max(decoder_output_poi[:,i,:i+1],dim=1).values
            pooled_cat[:,i]=torch.max(decoder_output_cat[:,i,:i+1],dim=1).values

        return pooled_poi,pooled_cat



