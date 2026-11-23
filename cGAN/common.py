import random

import numpy as np
import torch

N_BINS = 10 # số điều kiện (bin) của cGAN
IMG_SIZE = 256 # ảnh CXR xám 256x256

# Bộ COVID-QU-Ex: một nhãn nhị phân, 1 = covid19, 0 = normal
LABELS = ["covid19"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_index(name):
    if name not in LABELS:
        raise ValueError(f"Nhãn '{name}' không hợp lệ. Chọn một trong: {LABELS}")
    return LABELS.index(name)


# ánh xạ xác suất sang chỉ số bin
def to_bin(c):
    return (c.clamp(0, 1 - 1e-6) * N_BINS).long()


# giá trị tâm của bin, dùng làm target c cho KL
def bin_center(b):
    return (b.float() + 0.5) / N_BINS
