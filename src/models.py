"""Temporal neural networks.

Every model takes
  x_seq : (B, T, 35)  standardized space-weather history, oldest -> current
  x_cur : (B, C)      standardized current-time inputs: 8 static (+ NAIRAS for residual models)
and returns one standardized prediction per observation.
"""
import math

import torch
import torch.nn as nn


def head(n_in, hidden=64, p=0.1):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, 1))


class CurrentEncoder(nn.Module):
    """Small MLP for the current-time (static + NAIRAS) inputs."""
    def __init__(self, n_cur, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_cur, d), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class FlatMLP(nn.Module):
    """Whole history flattened into one vector: sees all the data, no notion of time."""
    def __init__(self, T, V, n_cur, width=64, p=0.1):
        super().__init__()
        w = 4 * width
        self.net = nn.Sequential(nn.Linear(T * V + n_cur, w), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(w, w // 2), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(w // 2, 1))

    def forward(self, x_seq, x_cur):
        return self.net(torch.cat([x_seq.flatten(1), x_cur], 1)).squeeze(-1)


class CNNMLP(nn.Module):
    """Hybrid CNN-MLP: 1D convolutions over time + current-time branch."""
    def __init__(self, T, V, n_cur, width=64, p=0.1):
        super().__init__()
        ch = width
        self.conv = nn.Sequential(nn.Conv1d(V, ch, 3, padding=1), nn.ReLU(), nn.Dropout(p),
                                  nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())
        self.cur = CurrentEncoder(n_cur)
        self.out = head(2 * ch + 32, p=p)

    def forward(self, x_seq, x_cur):
        h = self.conv(x_seq.transpose(1, 2))                     # B, ch, T
        z = torch.cat([h.mean(2), h[:, :, -1], self.cur(x_cur)], 1)
        return self.out(z).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation, p=0.1):
        super().__init__()
        self.pad = 2 * dilation                                   # causal: only look back
        self.c1 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        h = self.drop(torch.relu(self.c2(nn.functional.pad(h, (self.pad, 0)))))
        return torch.relu(x + h)


class TCN(nn.Module):
    """Temporal convolutional network: dilated causal convolutions (Bai et al. 2018)."""
    def __init__(self, T, V, n_cur, width=64, p=0.1):
        super().__init__()
        ch = width
        self.inp = nn.Conv1d(V, ch, 1)
        dil, d = [], 1
        while 2 * d < T or not dil:                               # receptive field covers the window
            dil.append(d); d *= 2
        self.blocks = nn.Sequential(*[TCNBlock(ch, k, p) for k in dil])
        self.cur = CurrentEncoder(n_cur)
        self.out = head(ch + 32, p=p)

    def forward(self, x_seq, x_cur):
        h = self.blocks(self.inp(x_seq.transpose(1, 2)))
        return self.out(torch.cat([h[:, :, -1], self.cur(x_cur)], 1)).squeeze(-1)


class AttentionBlock(nn.Module):
    """Pre-norm Transformer encoder block that also returns attention weights."""
    def __init__(self, d, heads, p=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, w                                               # w: B, T+1, T+1


class Transformer(nn.Module):
    """Each time step is a token; a learned summary token [S] attends over the history."""
    def __init__(self, T, V, n_cur, width=64, p=0.1, heads=4, layers=2):
        super().__init__()
        d = width
        self.embed = nn.Linear(V, d)
        self.pos = nn.Parameter(torch.randn(1, T + 1, d) * 0.02)
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([AttentionBlock(d, heads, p) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.cur = CurrentEncoder(n_cur)
        self.out = head(d + 32, p=p)
        self.last_attention = None

    def forward(self, x_seq, x_cur):
        B = x_seq.shape[0]
        h = torch.cat([self.summary.expand(B, -1, -1), self.embed(x_seq)], 1) + self.pos
        attn = []
        for blk in self.blocks:
            h, w = blk(h)
            attn.append(w[:, 0, 1:])                              # summary token -> each time step
        self.last_attention = torch.stack(attn, 1)                # B, layers, T
        z = torch.cat([self.norm(h[:, 0]), self.cur(x_cur)], 1)
        return self.out(z).squeeze(-1)


MODELS = {"FlatMLP": FlatMLP, "CNN-MLP": CNNMLP, "TCN": TCN, "Transformer": Transformer}
