import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

from common import DEVICE, N_BINS


def conv3(i, o):
    return nn.Conv2d(in_channels=i, out_channels=o, kernel_size=3, stride=1, padding=1)


def conv1(i, o):
    return nn.Conv2d(in_channels=i, out_channels=o, kernel_size=1)


class CondBN(nn.Module):
    def __init__(self, ch, n_bins):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch, affine=False)
        self.gamma, self.beta = nn.Embedding(n_bins, ch), nn.Embedding(n_bins, ch)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, y):
        return self.bn(x) * self.gamma(y)[:, :, None, None] + self.beta(y)[:, :, None, None]


class EncBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.bn1, self.c1 = nn.BatchNorm2d(i), conv3(i, o)
        self.bn2, self.c2 = nn.BatchNorm2d(o), conv3(o, o)
        self.sc = SN(conv1(i, o))

    def forward(self, x):
        h = self.c1(F.avg_pool2d(F.relu(self.bn1(x)), 2))
        h = self.c2(F.relu(self.bn2(h)))
        return h + self.sc(F.avg_pool2d(x, 2))


class GenBlock(nn.Module):
    def __init__(self, i, o, n_bins):
        super().__init__()
        self.bn1, self.c1 = CondBN(i, n_bins), conv3(i, o)
        self.bn2, self.c2 = CondBN(o, n_bins), conv3(o, o)
        self.sc = SN(conv1(i, o))

    def forward(self, x, y):
        h = F.interpolate(F.relu(self.bn1(x, y)), scale_factor=2, mode="nearest")
        h = self.c2(F.relu(self.bn2(self.c1(h), y)))
        return h + self.sc(F.interpolate(x, scale_factor=2, mode="nearest"))


class DiscBlock(nn.Module):
    def __init__(self, i, o, down=True, pre_relu=True):
        super().__init__()
        self.c1, self.c2, self.sc = SN(conv3(i, o)), SN(conv3(o, o)), SN(conv1(i, o))
        self.down, self.pre_relu = down, pre_relu

    def forward(self, x):
        h = self.c1(F.relu(x) if self.pre_relu else x)
        h = self.c2(F.relu(h))
        s = self.sc(x)
        if self.down:
            h, s = F.avg_pool2d(h, 2), F.avg_pool2d(s, 2)
        return h + s


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn, self.conv = nn.BatchNorm2d(1), conv3(1, 64)
        ch = [64, 128, 256, 512, 1024, 1024]
        self.blocks = nn.Sequential(*[EncBlock(ch[i], ch[i + 1]) for i in range(5)])

    def forward(self, x):
        return self.blocks(self.conv(F.relu(self.bn(x))))


class Generator(nn.Module):
    def __init__(self, n_bins=N_BINS):
        super().__init__()
        ch = [1024, 1024, 512, 256, 128, 64]
        self.blocks = nn.ModuleList([GenBlock(ch[i], ch[i + 1], n_bins) for i in range(5)])
        self.bn, self.out = nn.BatchNorm2d(64), conv3(64, 1)

    def forward(self, z, y):
        for b in self.blocks:
            z = b(z, y)
        return torch.tanh(self.out(F.relu(self.bn(z))))


class Discriminator(nn.Module):
    def __init__(self, n_bins=N_BINS):
        super().__init__()
        ch = [1, 64, 128, 256, 512, 1024]
        blocks = [DiscBlock(ch[i], ch[i + 1], down=True, pre_relu=(i > 0)) for i in range(5)]
        blocks.append(DiscBlock(1024, 1024, down=False))
        self.blocks = nn.Sequential(*blocks)
        self.fc = SN(nn.Linear(1024, 1))
        self.embed = SN(nn.Embedding(n_bins, 1024))
        self.n_bins = n_bins

    def forward(self, x, y):
        phi = F.relu(self.blocks(x)).sum(dim=(2, 3))  # global sum pooling
        r_x = self.fc(phi).squeeze(1)
        idx = torch.arange(self.n_bins, device=x.device)
        V = self.embed(idx) # (N,1024), đã spectral-norm
        proj = phi @ V.t() # (B,N): v_i^T phi(x)
        mask = (idx[None, :] < y[:, None]).float()  # chỉ cộng các i < n
        return r_x + (proj * mask).sum(1)


class Explainer(nn.Module):
    def __init__(self, n_bins=N_BINS):
        super().__init__()
        self.E, self.G = Encoder(), Generator(n_bins)

    def forward(self, x, y):
        return self.G(self.E(x), y)


def load_explainer(ckpt_path, device=DEVICE):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    X = Explainer()
    X.load_state_dict(ck["X"])
    return X.to(device).eval()
