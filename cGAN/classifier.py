import torch
import torch.nn as nn
from torchvision.models import densenet121
from common import DEVICE, LABELS, target_index

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(num_classes = len(LABELS)):
    m = densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


class TargetClassifier(nn.Module):
    def __init__(self, backbone, k= 0):
        super().__init__()
        self.backbone = backbone
        self.k = k
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode = True):
        return super().train(False) # không bật train mode

    def forward(self, x01):
        x = (x01.repeat(1, 3, 1, 1) - self.mean) / self.std
        return torch.sigmoid(self.backbone(x)[:, self.k])


def load_target_classifier(ckpt_path, target= "covid19", device=DEVICE):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    bb = build_backbone()
    bb.load_state_dict(state)
    return TargetClassifier(bb, target_index(target)).to(device)
