import torch
import torch.nn as nn


class SpectralDiffusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.t = nn.Parameter(torch.zeros(channels))

    def forward(self, x, evals, evecs, mass):
        coefs = evecs.T @ (mass[:, None] * x)
        decay = torch.exp(-evals[:, None] * self.t.abs()[None, :])
        return evecs @ (decay * coefs)


class DiffusionBlock(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.diffusion = SpectralDiffusion(channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )

    def forward(self, x, evals, evecs, mass):
        x_diffused = self.diffusion(x, evals, evecs, mass)
        return x + self.mlp(torch.cat([x, x_diffused], dim=-1))


class MiniDiffusionNet(nn.Module):
    def __init__(self, in_ch, num_classes, width=128, n_blocks=4, dropout=0.3):
        super().__init__()
        self.inp = nn.Linear(in_ch, width)
        self.blocks = nn.ModuleList(DiffusionBlock(width, dropout) for _ in range(n_blocks))
        self.out = nn.Linear(width, num_classes)

    def forward(self, x, evals, evecs, mass, faces):
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, evals, evecs, mass)
        vertex_logits = self.out(h)
        return vertex_logits[faces].mean(1)
