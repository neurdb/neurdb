"""
Bisection implementation of alpha-entmax (Peters et al., 2019).
Backward pass wrt alpha per (Correia et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.
"""

# Author: Goncalo M Correia
# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>

import torch
import torch.nn as nn
from torch.autograd import Function


class EntmaxBisectFunction(Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)

        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)

        f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):
            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

        return dX, d_alpha, None, None, None


class SparsemaxBisectFunction(EntmaxBisectFunction):
    @classmethod
    def _gp(cls, x, alpha):
        return x

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y

    @classmethod
    def _p(cls, x, alpha):
        return torch.clamp(x, min=0)

    @classmethod
    def forward(cls, ctx, X, dim=-1, n_iter=50, ensure_sum_one=True):
        return super().forward(ctx, X, alpha=2, dim=dim, n_iter=50, ensure_sum_one=True)

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors
        gppr = (Y > 0).to(dtype=dY.dtype)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None, None


def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)


def sparsemax_bisect(X, dim=-1, n_iter=50, ensure_sum_one=True):
    return SparsemaxBisectFunction.apply(X, dim, n_iter, ensure_sum_one)


class SparsemaxBisect(nn.Module):
    def __init__(self, dim=-1, n_iter=None):
        self.dim = dim
        self.n_iter = n_iter
        super().__init__()

    def forward(self, X):
        return sparsemax_bisect(X, dim=self.dim, n_iter=self.n_iter)


class EntmaxBisect(nn.Module):
    def __init__(self, alpha=1.5, dim=-1, n_iter=50):
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        super().__init__()

    def forward(self, X):
        return entmax_bisect(X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter)
