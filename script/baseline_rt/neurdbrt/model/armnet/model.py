import torch
import torch.nn as nn
from einops import rearrange

from .entmax import EntmaxBisect
from .layer import MLP, Embedding


class SparseAttLayer(nn.Module):
    def __init__(self, nhead, nfield, nemb, d_k, nhid, alpha=1.5):
        super(SparseAttLayer, self).__init__()
        self.sparsemax = (
            nn.Softmax(dim=-1) if alpha == 1.0 else EntmaxBisect(alpha, dim=-1)
        )
        self.scale = d_k**-0.5
        self.bilinear_w = nn.Parameter(torch.zeros(nhead, nemb, d_k))
        self.query = nn.Parameter(torch.zeros(nhead, nhid, d_k))
        self.values = nn.Parameter(torch.zeros(nhead, nhid, nfield))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bilinear_w, gain=1.414)
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        keys = x
        att_gates = (
            torch.einsum("bfx,kxy,koy->bkof", keys, self.bilinear_w, self.query)
            * self.scale
        )
        sparse_gates = self.sparsemax(att_gates)
        return torch.einsum("bkof,kof->bkof", sparse_gates, self.values)


class ARMNetModel(nn.Module):
    def __init__(
        self,
        nfield, nfeat, nemb, nhead, alpha, nhid,
        mlp_nlayer, mlp_nhid, dropout, ensemble,
        deep_nlayer, deep_nhid, noutput=1,
    ):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.attn_layer = SparseAttLayer(nhead, nfield, nemb, nemb, nhid, alpha)
        self.arm_bn = nn.BatchNorm1d(nhead * nhid)
        self.mlp = MLP(nhead * nhid * nemb, mlp_nlayer, mlp_nhid, dropout, noutput=noutput)
        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(nfield * nemb, deep_nlayer, deep_nhid, dropout, noutput=noutput)
            self.ensemble_layer = nn.Linear(2 * noutput, 1 * noutput)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.0)

    def forward(self, x):
        x["value"].clamp_(0.001, 1.0)
        x_arm = self.embedding(x)
        arm_weight = self.attn_layer(x_arm)
        x_arm = torch.exp(torch.einsum("bfe,bkof->bkoe", x_arm, arm_weight))
        x_arm = rearrange(x_arm, "b k o e -> b (k o) e")
        if x_arm.shape[0] > 1:
            x_arm = self.arm_bn(x_arm)
        x_arm = rearrange(x_arm, "b h e -> b (h e)")
        y = self.mlp(x_arm)
        if hasattr(self, "ensemble_layer"):
            x_deep = self.deep_embedding(x)
            x_deep = rearrange(x_deep, "b f e -> b (f e)")
            y_deep = self.deep_mlp(x_deep)
            y = torch.cat([y, y_deep], dim=1)
            y = self.ensemble_layer(y)
        return y.squeeze(1)
