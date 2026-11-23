import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from classifier import load_target_classifier
from common import DEVICE, N_BINS, bin_center, seed_everything, to_bin
from data import CXRDataset, balance_by_bin, make_balanced_sampler, score_to_bin_np
from losses import (classifier_consistency_loss, d_loss, g_adv_loss,
                    reconstruction_loss_l1)
from models import Discriminator, Explainer
from visualize import progressive_counterfactuals, save_strip_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True, help="File CSV có cột `score`.")
    p.add_argument("--classifier_path", required=True, help = "Đường dẫn file checkpoint mô hình phân loại.")
    p.add_argument("--target", default="covid19", help = "Tên nhãn mục tiêu cần lấy xác suất.")
    p.add_argument("--out_dir", default="runs/exp")
    p.add_argument("--iters", type=int, default=50000, help="Tổng số bước D (G/E cập nhật iters/n_critic lần)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--per_bin", type=int, default=3000, help="Số ảnh tối đa mỗi bin")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n_critic", type=int, default=5, help = "Số bước cập nhật Discriminator trước mỗi lần cập nhật Generator.")
    p.add_argument("--lam_gan", type=float, default=1.0, help = "Hệ số cho loss GAN.")
    p.add_argument("--lam_f", type=float, default=1.0, help = "Hệ số cho loss consistency classifier.")
    p.add_argument("--lam_rec", type=float, default=0.5, help = "Hệ số cho loss reconstruction.")
    p.add_argument("--log_every", type=int, default=50, help="Tần suất log thông tin huấn luyện (tính theo số bước G).")
    p.add_argument("--save_every", type=int, default=5000, help = "Số bước lặp để lưu checkpoint mô hình.")
    p.add_argument("--sample_every", type=int, default=2500, help = "Số bước lặp để sinh ảnh minh họa kiểm tra tiến độ học.")
    p.add_argument("--resume", default=None, help="Đường dẫn checkpoint để train tiếp")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    a = parse_args()
    seed_everything(a.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out_dir, exist_ok=True)

    # dữ liệu: giới hạn mỗi bin và lấy mẫu cân bằng
    df = balance_by_bin(pd.read_csv(a.csv_path), a.per_bin, a.seed)
    counts = pd.Series(score_to_bin_np(df["score"].values)).value_counts().sort_index()
    print("Số ảnh mỗi bin:", counts.to_dict(), flush=True)
    if len(counts) < N_BINS:
        print("CẢNH BÁO: có bin rỗng -> cGAN sẽ không học được điều kiện ở bin đó.")
    loader = DataLoader(CXRDataset(df, "score"), batch_size=a.batch_size, sampler=make_balanced_sampler(df),
                        num_workers=a.num_workers, drop_last=True, pin_memory=True)
    batches = infinite(loader)

    # mô hình
    f = load_target_classifier(a.classifier_path, a.target)
    X, D = Explainer().to(DEVICE), Discriminator().to(DEVICE)
    opt_g = torch.optim.Adam(X.parameters(), lr=a.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.0, 0.9))

    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        X.load_state_dict(ck["X"])
        D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        start = ck["it"]
        print(f"Resume từ iter {start}")

    def save(path, it):
        torch.save({"X": X.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "it": it, "args": vars(a)}, path)

    fixed_x01 = next(batches)[0][:4].to(DEVICE) # batch cố định để xem tiến triển

    stats = {}
    for it in range(start + 1, a.iters + 1):
        x01, fx = next(batches)
        x01, fx = x01.to(DEVICE, non_blocking=True), fx.to(DEVICE, non_blocking=True)
        x = x01 * 2 - 1 # GAN làm việc trong [-1,1]
        B = x.size(0)
        bin_x = to_bin(fx) # condition thật của ảnh thật = bin của f(x)
        bin_c = to_bin(torch.rand(B, device=DEVICE)) # condition mục tiêu ngẫu nhiên
        c_target = bin_center(bin_c)

        # bước D 
        with torch.no_grad():
            x_fake = X(x, bin_c)
        l_d = d_loss(D(x, bin_x), D(x_fake, bin_c))
        opt_d.zero_grad(set_to_none=True)
        l_d.backward()
        opt_d.step()
        stats["D"] = l_d.item()

        # bước G + E
        if it % a.n_critic == 0:
            z = X.E(x)
            x_c = X.G(z, bin_c) # explanation
            l_adv = g_adv_loss(D(x_c, bin_c)) # (A) data consistency
            l_f = classifier_consistency_loss(f((x_c + 1) / 2), c_target) # (B) classifier consistency
            x_self = X.G(z, bin_x) # self-reconstruction
            x_cyc = X.G(X.E(x_c), bin_x) # cyclic reconstruction
            l_rec = reconstruction_loss_l1(x, x_self, x_cyc) # (C) self-consistency (L1)
            l_g = a.lam_gan * l_adv + a.lam_f * l_f + a.lam_rec * l_rec
            opt_g.zero_grad(set_to_none=True)
            l_g.backward()
            opt_g.step()
            stats.update(adv=l_adv.item(), kl=l_f.item(), rec=l_rec.item())

            if (it // a.n_critic) % a.log_every == 0:
                print(f"iter {it}/{a.iters} | " + " ".join(f"{k} {v:.4f}" for k, v in stats.items()), flush=True)

        if it % a.sample_every == 0:
            X.eval()
            xcs, preds, fxs = progressive_counterfactuals(X, f, fixed_x01)
            for i in range(xcs.size(0)):
                save_strip_png(os.path.join(a.out_dir, f"sample_it{it}_{i}.png"),
                               fixed_x01[i], xcs[i], preds[i], fxs[i].item())
            X.train()

        if it % a.save_every == 0 or it == a.iters:
            save(os.path.join(a.out_dir, "explainer_last.pt"), it)

    print("Xong. Checkpoint:", os.path.join(a.out_dir, "explainer_last.pt"))


if __name__ == "__main__":
    main()
