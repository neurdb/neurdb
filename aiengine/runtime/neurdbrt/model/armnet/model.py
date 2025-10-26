import torch
import torch.nn as nn
from einops import rearrange

from .entmax import EntmaxBisect
from .layer import MLP, Embedding


class SparseAttLayer(nn.Module):
    def __init__(
        self,
        nhead: int,
        nfield: int,
        nemb: int,
        d_k: int,
        nhid: int,
        alpha: float = 1.5,
    ):
        """Multi-Head Sparse Attention Layer"""
        super(SparseAttLayer, self).__init__()
        self.sparsemax = (
            nn.Softmax(dim=-1) if alpha == 1.0 else EntmaxBisect(alpha, dim=-1)
        )

        self.scale = d_k**-0.5
        self.bilinear_w = nn.Parameter(torch.zeros(nhead, nemb, d_k))  # nhead*nemb*d_k
        self.query = nn.Parameter(torch.zeros(nhead, nhid, d_k))  # nhead*nhid*d_k
        self.values = nn.Parameter(
            torch.zeros(nhead, nhid, nfield)
        )  # nhead*nhid*nfield
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.bilinear_w, gain=1.414)
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        @param x:   [bsz, nfield, nemb], FloatTensor
        :return:    Att_weights [bsz, nhid, nfield], FloatTensor
        """
        keys = x  # bsz*nfield*nemb
        # sparse gates
        att_gates = (
            torch.einsum("bfx,kxy,koy->bkof", keys, self.bilinear_w, self.query)
            * self.scale
        )  # bsz*nhead*nhid*nfield
        sparse_gates = self.sparsemax(att_gates)  # bsz*nhead*nhid*nfield
        return torch.einsum("bkof,kof->bkof", sparse_gates, self.values)


class ARMNetModel(nn.Module):
    """
    Update one layer of the model in the database
    @param model_id: the model id
    @param layer_id: the layer id
    @param layer_data: the new layer data in bytes
    @return: None
    """

    def __init__(
        self,
        nfield: int,
        nfeat: int,
        nemb: int,
        nhead: int,
        alpha: float,
        nhid: int,
        mlp_nlayer: int,
        mlp_nhid: int,
        dropout: float,
        ensemble: bool,
        deep_nlayer: int,
        deep_nhid: int,
        noutput: int = 1,
    ):
        """
        @param nfield:          Number of Fields
        @param nfeat:           Total Number of Features
        @param nemb:            Feature Embedding size
        @param nhead:           Number of Attention Heads (each with a bilinear attn weight)
        @param alpha:           Sparsity hyper-parameter for ent-max
        @param nhid:            Number of Exponential Neuron
        @param mlp_nlayer:      Number of layers for prediction head
        @param mlp_nhid:        Number of hidden neurons for prediction head
        @param dropout:         Dropout rate
        @param ensemble:        Whether to Ensemble with a DNN
        @param deep_nlayer:     Number of layers for Ensemble DNN
        @param deep_nhid:       Number of hidden neurons for Ensemble DNN
        @param noutput:         Number of prediction output, e.g., 1 for binary cls
        """
        super().__init__()
        # embedding
        self.embedding = Embedding(nfeat, nemb)
        # arm
        self.attn_layer = SparseAttLayer(nhead, nfield, nemb, nemb, nhid, alpha)
        self.arm_bn = nn.BatchNorm1d(nhead * nhid)
        # MLP
        self.mlp = MLP(
            nhead * nhid * nemb, mlp_nlayer, mlp_nhid, dropout, noutput=noutput
        )
        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(
                nfield * nemb, deep_nlayer, deep_nhid, dropout, noutput=noutput
            )
            self.ensemble_layer = nn.Linear(2 * noutput, 1 * noutput)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.0)

    def forward(self, x):
        """
        @param x:   {'id': [bsz, nfield], LongTensor, 'value': [bsz, nfield], FloatTensor}
        :return:    y: [bsz], FloatTensor of size B, for Regression or Classification
        """
        x["value"].clamp_(0.001, 1.0)
        x_arm = self.embedding(x)  # bsz*nfield*nemb

        arm_weight = self.attn_layer(x_arm)  # bsz*nhead*nhid*nfield
        x_arm = torch.exp(
            torch.einsum("bfe,bkof->bkoe", x_arm, arm_weight)
        )  # bsz*nhead*nhid*nemb
        x_arm = rearrange(x_arm, "b k o e -> b (k o) e")  # bsz*(nhead*nhid)*nemb
        
        if x_arm.shape[0] > 1: # apply batch norm only when batch size > 1
            x_arm = self.arm_bn(x_arm)  # bsz*(nhead*nhid)*nemb
        
        x_arm = rearrange(x_arm, "b h e -> b (h e)")  # bsz*(nhead*nhid*nemb)

        y = self.mlp(x_arm)  # bsz*noutput
        if hasattr(self, "ensemble_layer"):
            x_deep = self.deep_embedding(x)  # bsz*nfield*nemb
            x_deep = rearrange(x_deep, "b f e -> b (f e)")  # bsz*(nfield*nemb)
            y_deep = self.deep_mlp(x_deep)  # bsz*noutput

            y = torch.cat([y, y_deep], dim=1)  # bsz*(2*noutput)
            y = self.ensemble_layer(y)  # bsz*noutput
        
        y = y.squeeze(1)  # bsz*noutput

        return y
