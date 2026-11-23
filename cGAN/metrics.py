import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3

from common import DEVICE, LABELS

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# CV
def counterfactual_flip(f_x, f_xc, tau = 0.8):
    return (f_x - f_xc).abs() > tau


# Inception (FID)
def build_inception_classifier(pretrained = True, num_classes = len(LABELS)):
    if pretrained:
        m = inception_v3(weights="DEFAULT") # init ImageNet
        m.aux_logits, m.AuxLogits = False, None
        m.fc = nn.Linear(2048, num_classes)
    else:
        m = inception_v3(weights=None, aux_logits=False, num_classes=num_classes, transform_input=True)
    return m


def build_fid_net(ckpt_path = None, device=DEVICE):
    if ckpt_path:
        m = build_inception_classifier(pretrained=False)
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        m = inception_v3(weights="DEFAULT")
        m.aux_logits, m.AuxLogits = False, None
    m.fc = nn.Identity()
    return m.to(device).eval()


def prepare_for_inception(x01):
    x = F.interpolate(x01.repeat(1, 3, 1, 1), size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def inception_features(net, x01):
    return net(prepare_for_inception(x01))


def activation_stats(feats):
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2, eps = 1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all(): # ma trận gần suy biến
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


def fid_from_features(real_feats, fake_feats):
    return frechet_distance(*activation_stats(real_feats), *activation_stats(fake_feats))
