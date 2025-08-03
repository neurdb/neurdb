import numpy as np

from auto_pipeline import deterministic


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.lp_buffer = []

    def add(self, s0, a, r, s1, done, index, fixline_id, ctx):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(
            (
                s0[None, :],
                a,
                r,
                s1[None, :],
                done,
                index,
                fixline_id,
                ctx.detach().numpy(),
            )
        )

    def sample(self, batch_size):
        s0, a, r, s1, done, index, fixline_id, ctx = zip(
            *deterministic.buffer_rng.choice(self.buffer, batch_size, replace=False)
        )
        return (
            np.concatenate(s0),
            a,
            r,
            np.concatenate(s1),
            done,
            index,
            fixline_id,
            ctx,
        )

    def lp_add(self, s0, a, r, ctx):
        if len(self.lp_buffer) >= self.capacity:
            self.lp_buffer.pop(0)
        self.lp_buffer.append((s0[None, :], a, r, ctx))

    def lp_sample(self, batch_size):
        s0, a, r, ctx = zip(
            *deterministic.buffer_rng.choice(self.lp_buffer, batch_size, replace=False)
        )
        return np.concatenate(s0), a, r, ctx

    def size(self):
        return len(self.buffer)

    def lp_size(self):
        return len(self.lp_buffer)
