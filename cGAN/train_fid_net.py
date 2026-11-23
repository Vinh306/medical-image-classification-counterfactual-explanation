import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import DEVICE, seed_everything
from data import CXRDataset
from metrics import build_inception_classifier, prepare_for_inception


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="Đường dẫn file CSV chứa dữ liệu huấn luyện.")
    p.add_argument("--out", default="ckpt/inception_cxr.pt", help="Đường dẫn lưu file checkpoint của mô hình Inception.")
    p.add_argument("--epochs", type=int, default=1, help="Số epoch huấn luyện.")
    p.add_argument("--batch_size", type=int, default=32, help="Kích thước batch.")
    p.add_argument("--lr", type=float, default=1e-4, help="Tốc độ học.")
    p.add_argument("--max_iters", type=int, default=None, help="Số lần lặp tối đa.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    seed_everything(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dl = DataLoader(CXRDataset(pd.read_csv(a.train_csv), "labels"), batch_size=a.batch_size, shuffle=True,
                    num_workers=a.num_workers, drop_last=True, pin_memory=True)
    net = build_inception_classifier(pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    it = 0
    for ep in range(a.epochs):
        net.train()
        for x, y in dl:
            logits = net(prepare_for_inception(x.to(DEVICE)))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            if it % 100 == 0:
                print(f"epoch {ep} iter {it} loss {loss.item():.4f}", flush=True)
            if a.max_iters and it >= a.max_iters:
                break
        torch.save(net.state_dict(), a.out)
        if a.max_iters and it >= a.max_iters:
            break
    print("Đã lưu", a.out)


if __name__ == "__main__":
    main()
