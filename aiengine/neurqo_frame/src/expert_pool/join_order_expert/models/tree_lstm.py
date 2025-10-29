import torch.nn as nn
import torch

DEVICE = torch.cuda.is_available()


class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fc_left = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_right = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_input = nn.Linear(input_size, 5 * hidden_size)
        elementwise_affine = False
        self.layer_norm_input = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)
        self.layer_norm_left = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)
        self.layer_norm_right = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)
        self.layer_norm_c = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, h_left, c_left, h_right, c_right, feature):
        lstm_in = self.layer_norm_left(self.fc_left(h_left))
        lstm_in += self.layer_norm_right(self.fc_right(h_right))
        lstm_in += self.layer_norm_input(self.fc_input(feature))
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * c_left +
             f2.sigmoid() * c_right)
        c = self.layer_norm_c(c)
        h = o.sigmoid() * c.tanh()
        return h, c

    def zero_h_c(self, input_dim=1):
        return torch.zeros(input_dim, self.hidden_size, device=DEVICE),\
               torch.zeros(input_dim, self.hidden_size,
                                                                                    device=DEVICE)
