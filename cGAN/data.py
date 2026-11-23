import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from common import IMG_SIZE, LABELS, N_BINS


def load_image(path):
    img = Image.open(path).convert("L")
    w, h = img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


class CXRDataset(Dataset):
    def __init__(self, df, mode = "none"):
        assert mode in ("none", "labels", "score")
        self.paths = df["path"].tolist()
        if mode == "labels":
            lab = df[LABELS].astype(float).fillna(0.0).values
            self.y = np.where(lab == -1, 1.0, lab).astype(np.float32)
        elif mode == "score":
            self.y = df["score"].values.astype(np.float32)
        else:
            self.y = np.zeros(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i]), torch.as_tensor(self.y[i])


# chuyển xác suất thành chỉ số bin
def score_to_bin_np(score):
    return np.minimum((score * N_BINS).astype(int), N_BINS - 1)


# lấy số ảnh cân bằng giữa các bin
def balance_by_bin(df, per_bin, seed = 0):
    bins = score_to_bin_np(df["score"].values)
    parts = []
    for b in range(N_BINS):
        sub = df[bins == b]
        if len(sub) > per_bin:
            sub = sub.sample(per_bin, random_state=seed)
        parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


# mỗi batch có xấp xỉ đều các bin
def make_balanced_sampler(df):
    bins = score_to_bin_np(df["score"].values)
    counts = np.bincount(bins, minlength=N_BINS)
    weights = 1.0 / counts[bins]
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True)
