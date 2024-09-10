import torch
import torch.nn as nn


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


class MLP(nn.Module):
    def __init__(self, ninput, nlayers, nhid, dropout, noutput=1):
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
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


class DNNModel(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """

    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.mlp_ninput = nfield * nemb
        self.mlp = MLP(self.mlp_ninput, mlp_layers, mlp_hid, dropout)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)  # B*F*E
        y = self.mlp(x_emb.view(-1, self.mlp_ninput))  # B*1
        return y.squeeze(1)
