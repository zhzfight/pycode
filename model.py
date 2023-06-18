import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from math import pi, log

from einops import rearrange, repeat


class PoiEmbeddings(nn.Module):
    def __init__(self,num_pois,embedding_dim):
        super(PoiEmbeddings,self).__init__()
        self.poi_embedding=nn.Embedding(num_embeddings=num_pois,embedding_dim=embedding_dim)
    def forward(self,poi_idx):
        return self.poi_embedding(poi_idx)

class TimeEmbeddings(nn.Module):
    def __init__(self,embedding_dim):
        super(TimeEmbeddings,self).__init__()
        self.time_embedding=nn.Embedding(num_embeddings=24*7,embedding_dim=embedding_dim)
    def forward(self,time_idx):
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
    def __init__(self, num_poi, num_cat, nhid,batch_size, device,dropout,user_dim):
        super(TimeIntervalAwareTransformer, self).__init__()


        self.device=device
        self.nhid=nhid
        self.batch_size=batch_size
        # self.encoder = nn.Embedding(num_poi, embed_size)

        self.decoder_poi = nn.Linear(nhid, num_poi)

        self.day_embedding=nn.Embedding(8,nhid,padding_idx=0)
        self.hour_embedding=nn.Embedding(26,nhid,padding_idx=0)

        self.label_day_embedding=nn.Embedding(8,nhid,padding_idx=0)
        self.label_hour_embedding = nn.Embedding(26, nhid, padding_idx=0)

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
        self.rotary_emb_attn = RotaryEmbedding(dim=nhid)
        self.rotary_emb_decode=RotaryEmbedding(dim=nhid)
        self.u_proj=nn.Linear(user_dim,nhid)




    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, batch_seq_lens, batch_input_seqs_h,batch_input_seqs_w,batch_label_seqs_h,batch_label_seqs_w,batch_user_embedding):
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
                        hourInterval[i][j][k]=abs(batch_input_seqs_h[i][j]-batch_input_seqs_h[i][k])+2
                    dayInterval[i][j][k]=abs(batch_input_seqs_w[i][j]-batch_input_seqs_w[i][k])+1

                    label_hourInterval[i][j][k] = abs(batch_label_seqs_h[i][j]-batch_input_seqs_h[i][k]) + 2
                    label_dayInterval[i][j][k] = abs(batch_label_seqs_w[i][j]-batch_input_seqs_w[i][k]) + 1


        hourInterval_embedding=self.hour_embedding(hourInterval)
        dayInterval_embedding=self.day_embedding(dayInterval)

        label_hourInterval_embedding=self.label_hour_embedding(label_hourInterval)
        label_dayInterval_embedding=self.label_day_embedding(label_dayInterval)

        # mask attn
        attn_mask = ~torch.tril(torch.ones((src.shape[1], src.shape[1]), dtype=torch.bool, device=self.device))
        time_mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool, device=self.device)
        for i in range(src.shape[0]):
            time_mask[i, batch_seq_lens[i]:] = True

        attn_mask = attn_mask.unsqueeze(0).expand(src.shape[0], -1, -1)
        time_mask = time_mask.unsqueeze(-1).expand(-1, -1, src.shape[1])

        src=self.rotary_emb_attn.rotate_queries_or_keys(src)
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


        #attn_mask=attn_mask.unsqueeze(-1).expand(-1,-1,-1,ffn_output.shape[-1])
        ffn_output=ffn_output.unsqueeze(2).repeat(1,1,ffn_output.shape[1],1).transpose(2,1)
        ffn_output=self.rotary_emb_decode.rotate_queries_or_keys(ffn_output)
        ffn_output=torch.add(ffn_output,label_hourInterval_embedding)
        ffn_output=torch.add(ffn_output,label_dayInterval_embedding)
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
        pooled_poi = self.decoder(aggregated_preference)
        return pooled_poi


def exists(val):
    return val is not None

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1.):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)

# learned rotation helpers



# classes

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache_scale = dict()
        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.use_xpos = use_xpos
        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.register_buffer('scale', scale)

    def rotate_queries_or_keys(self, t, seq_dim = -2):
        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
        device, seq_len = t.device, t.shape[seq_dim]
        freqs = self.forward(lambda: torch.arange(seq_len, device = device), cache_key = seq_len)
        return apply_rotary_emb(freqs, t)


    def forward(self, t, cache_key = None):
        if callable(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)


        return freqs

