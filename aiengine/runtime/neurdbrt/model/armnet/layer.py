import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    embeddings B*F*E
        """
        emb = self.embedding(x["id"])  # B*F*E
        return emb * x["value"].unsqueeze(2)  # B*F*E


class Linear(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        self.weight = nn.Embedding(nfeat, 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    linear transform of x
        """
        linear = self.weight(x["id"]).squeeze(2) * x["value"]  # B*F
        return torch.sum(linear, dim=1) + self.bias  # B


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_dim=True):
        super().__init__()
        self.reduce_dim = reduce_dim

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        """
        square_of_sum = torch.sum(x, dim=1) ** 2  # B*E
        sum_of_square = torch.sum(x**2, dim=1)  # B*E
        fm = square_of_sum - sum_of_square  # B*E
        if self.reduce_dim:
            fm = torch.sum(fm, dim=1)  # B
        return 0.5 * fm  # B*E/B


def get_triu_indices(n, diag_offset=1):
    """get the row, col indices for the upper-triangle of an (n, n) array"""
    return np.triu_indices(n, diag_offset)


def get_all_indices(n):
    """get all the row, col indices for an (n, n) array"""
    return map(list, zip(*[(i, j) for i in range(n) for j in range(n)]))


class MLP(nn.Module):
    def __init__(self, ninput, nlayers, nhid, dropout, noutput=1):
        super().__init__()
        layers = list()

        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            # layers.append(nn.BatchNorm1d(nhid)) # TODO: find a better way for non-batched data
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid

        if nlayers == 0:
            nhid = ninput

        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)


def normalize_adj(adj):
    """normalize and return a adjacency matrix (numpy array)"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


class SelfAttnLayer(nn.Module):
    def __init__(self, nemb):
        """Self Attention Layer (scaled dot-product)"""
        super(SelfAttnLayer, self).__init__()
        self.Wq = nn.Linear(nemb, nemb, bias=False)
        self.Wk = nn.Linear(nemb, nemb, bias=False)
        self.Wv = nn.Linear(nemb, nemb, bias=False)

    def forward(self, x):
        """
        :param x:   B*F*E
        :return:    B*F*E
        """
        query, key, value = self.Wq(x), self.Wk(x), self.Wv(x)
        d_k = query.size(-1)
        scores = torch.einsum("bxe,bye->bxy", query, key)  # B*F*F
        attn_weights = F.softmax(scores / math.sqrt(d_k), dim=-1)  # B*F*F
        return torch.einsum("bxy,bye->bxe", attn_weights, value), attn_weights


class scaled_dot_prodct_attention_(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head Attention Module"""

    def __init__(self, nhead, ninput, n_k, n_v, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.nhead, self.n_k, self.n_v = nhead, n_k, n_v

        self.Wq = nn.Linear(ninput, nhead * n_k, bias=False)
        self.Wk = nn.Linear(ninput, nhead * n_k, bias=False)
        self.Wv = nn.Linear(ninput, nhead * n_v, bias=False)

        self.attn_layer = scaled_dot_prodct_attention_(temperature=n_k**0.5)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(ninput, eps=1e-6)

        self.fc = nn.Linear(nhead * n_v, ninput, bias=False)

    def forward(self, x, mask=None):
        """
        :param x:       B*F*E
        :param mask:    B*F*F
        :return:        B*F*E
        """
        bsz, seq_len = x.size(0), x.size(1)
        residual = x

        query = self.Wq(x).view(bsz, seq_len, self.nhead, self.n_k)  # B*F*H*Ek
        key = self.Wk(x).view(bsz, seq_len, self.nhead, self.n_k)  # B*F*H*Ek
        value = self.Wv(x).view(bsz, seq_len, self.nhead, self.n_v)  # B*F*H*Ev

        q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # B*1*N*N

        y, attn = self.attn_layer(q, k, v, mask=mask)  # B*H*F*Ev,

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)  # B*F*(HxEv)
        y = self.dropout(self.fc(y))  # B*F*E
        y += residual
        y = self.layer_norm(y)  # B*F*E
        return y, attn
