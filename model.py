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


class MeanAggregator1(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings and transform
    """
    def __init__(self, id2feat,  device,feature_dim,embed_dim,num_sample,dropout):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(MeanAggregator1, self).__init__()
        self.id2feat = id2feat
        self.device=device
        self.num_sample=num_sample
        self.W=nn.Linear(feature_dim,embed_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, nodes, to_neighs):
        """
        nodes --- list of nodes in a batch
        dis --- shape alike adj
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # node n
        # to_neighs n * ??
        samp_neighs = []
        for i, to_neigh in enumerate(to_neighs):
            if len(to_neigh) > self.num_sample:
                samp_neighs.append(random.sample(to_neigh, self.num_sample))
            else:
                samp_neighs.append(to_neigh)


        # ignore the unlinked nodes
        tmp = []
        for samp_neigh in samp_neighs:
            for item in samp_neigh:
                tmp.append(item)

        unique_nodes_list = list(set(tmp))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes))).to(self.device)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]


        mask[row_indices, column_indices] = 1

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)


        embed_matrix = self.id2feat[torch.LongTensor(unique_nodes_list).to(self.device)]  # （unique_count, feat_dim)
        embed_matrix=self.W(embed_matrix)
        to_feats = mask.mm(embed_matrix)  # n * embed_dim
        return to_feats  # n * embed_dim




class AttnAggregator1(nn.Module):
    def __init__(self, id2feat, device, feature_dim, embed_dim,num_sample,dropout):
        super(AttnAggregator1,self).__init__()
        self.id2feat = id2feat
        self.device=device
        self.num_sample = num_sample
        self.W=nn.Linear(feature_dim,embed_dim)
        self.W_Q = nn.Linear(feature_dim, embed_dim)
        self.W_K = nn.Linear(feature_dim, embed_dim)
        self.W_V = nn.Linear(feature_dim, embed_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self, nodes, to_neighs):
        # nodes n
        # to_neighs n * ??
        samp_neighs = []
        for i, to_neigh in enumerate(to_neighs):
            if len(to_neigh) > self.num_sample:
                samp_neighs.append(random.sample(to_neigh, self.num_sample))
            else:
                samp_neighs.append(to_neigh)


        tmp=[]
        for samp_neigh in samp_neighs:
            tmp.append(self.id2feat[torch.LongTensor(samp_neigh).to(self.device)])
        tmp=pad_sequence(tmp,batch_first=True,padding_value=0) # n * L * feature_dim

        self_feats=self.id2feat[nodes] # n * feature_dim
        Q=self.W_Q(self_feats) # n * embed_dim
        K=self.W_K(torch.cat((tmp,self_feats.unsqueeze(1)),dim=1)) # n * L *embed_dim
        V=self.W_V(torch.cat((tmp,self_feats.unsqueeze(1)),dim=1))

        attn_score=torch.bmm(Q.unsqueeze(1),K.transpose(2,1)).squeeze(1) # n * L
        mask = torch.zeros_like(attn_score).bool()
        for samp_neigh in samp_neighs:
            mask[:,len(samp_neigh):]=True

        attn_score = F.softmax(attn_score.masked_fill(mask, float('-inf')), dim=-1)
        mix_feats = torch.matmul(attn_score.unsqueeze(1), V).squeeze(1) # (n * 1 * L) (n * L * embed_dim) = (n * 1 * embed_dim)


        return mix_feats

class SageLayer1(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    id2feat -- function mapping LongTensor of node ids to FloatTensor of feature values.
    cuda -- whether to use GPU
    gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
    """
    def __init__(self, id2feat, adj_list, dis_list,num_sample ,feature_dim, embed_dim, device,dropout):
        super(SageLayer1, self).__init__()
        self.id2feat = id2feat
        self.dropout=dropout

        self.Meanagg = MeanAggregator1(self.id2feat, device,feature_dim,embed_dim,num_sample,dropout)
        self.Attnagg = AttnAggregator1(self.id2feat,device,feature_dim,embed_dim,num_sample,dropout)
        self.num_sample = num_sample
        self.device=device
        self.adj_list = adj_list
        self.dis_list=dis_list
        self.WC=nn.Linear(2*embed_dim,embed_dim)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        context_feats = self.Meanagg(nodes, [self.dis_list[int(node)] for node in nodes])
        mix_feats=self.Attnagg(nodes,[self.adj_list[int(node)] for node in nodes])

        combined = torch.cat((mix_feats,context_feats), dim=-1)  # (?, 2*feat_dim)
        combined=F.tanh(self.WC(combined))
        combined=F.normalize(combined,p=2,dim=-1)
        return combined


class AttnAggregator2(nn.Module):
    def __init__(self,id2feat,device,embed_dim,num_sample,dropout):
        super(AttnAggregator2,self).__init__()
        self.id2feat=id2feat
        self.num_sample=num_sample
        self.device=device
        self.W_Q=nn.Linear(embed_dim,embed_dim)
        self.W_K=nn.Linear(embed_dim,embed_dim)
        self.W_V=nn.Linear(embed_dim,embed_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, node, to_neighs):

        self_feats=self.id2feat([node])
        if len(to_neighs) > self.num_sample:
            to_neighs=random.sample(to_neighs, self.num_sample)

        samp_neighs=self.id2feat(torch.LongTensor(to_neighs).to(self.device))

        Q=self.W_Q(self_feats)   # 1 * embed_dim
        K=self.W_K(torch.cat((self_feats,samp_neighs),dim=0))  # neighs * embed_dim
        V=self.W_V(torch.cat((self_feats,samp_neighs),dim=0))  # neighs * embed_dim
        attn_score=torch.mm(Q,K.transpose(1,0)) # 1 * neighs
        attn_score = F.softmax(attn_score,dim=-1) # 1 * neighs
        mix_feats=torch.matmul(attn_score, V) # 1 * embed_dim

        return mix_feats
class SageLayer2(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    id2feat -- function mapping LongTensor of node ids to FloatTensor of feature values.
    cuda -- whether to use GPU
    gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
    """
    def __init__(self, id2feat, adj_list, num_sample, embed_dim, device,dropout):
        super(SageLayer2, self).__init__()
        self.id2feat = id2feat
        self.agg = AttnAggregator2(self.id2feat, device, embed_dim,num_sample,dropout)
        self.num_sample = num_sample
        self.device=device
        self.adj_list = adj_list
    def forward(self, node):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """

        feats= self.agg(node, self.adj_list[node])
        feats=F.tanh(feats)
        feats=F.normalize(feats,p=2,dim=-1)

        return feats


class GraphSage(nn.Module):
    def __init__(self, X, num_node, num_sample, embed_dim, adj, dis, device,dropout):
        super(GraphSage, self).__init__()
        self.id2node = X
        self.device=device
        self.layer1 = SageLayer1(self.id2node, adj, dis, num_sample, self.id2node.shape[1], embed_dim, device,dropout)
        self.layer2 = SageLayer2(lambda nodes: self.layer1(nodes), adj, num_sample, embed_dim, device,dropout)


    def forward(self, nodes):
        neigh_embeds = self.layer2(nodes)

        return neigh_embeds

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
        self.decoder_time = nn.Linear(embed_size, 1)
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
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)


        return out_poi, out_time, out_cat
