import argparse
import json
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
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
    p = argparse.ArgumentParser(description="Huấn luyện mô hình cGAN (Singla et al., 2022)")
    p.add_argument("--csv_path", required=True, help="File CSV chứa cột `image_path` và `score`.")
    p.add_argument("--classifier_path", required=True, help="Đường dẫn file checkpoint mô hình phân loại.")
    p.add_argument("--target", default="covid19", help="Tên nhãn mục tiêu cần lấy xác suất.")
    p.add_argument("--out_dir", default="runs/exp", help="Thư mục lưu kết quả và checkpoint.")
    p.add_argument("--iters", type=int, default=50000, help="Tổng số bước D (G/E cập nhật iters/n_critic lần).")
    p.add_argument("--batch_size", type=int, default=16, help="Kích thước batch huấn luyện.")
    p.add_argument("--per_bin", type=int, default=3000, help="Số ảnh tối đa mỗi bin.")
    p.add_argument("--lr", type=float, default=2e-4, help="Tốc độ học Adam.")
    p.add_argument("--n_critic", type=int, default=5, help="Số bước D trước mỗi bước G/E.")
    p.add_argument("--lam_gan", type=float, default=1.0, help="Hệ số cho loss GAN đối kháng (data consistency).")
    p.add_argument("--lam_f", type=float, default=1.0, help="Hệ số cho loss consistency classifier.")
    p.add_argument("--lam_rec", type=float, default=0.5, help="Hệ số cho loss tái tạo L1 (self & cyclic).")
    p.add_argument("--log_every", type=int, default=50, help="Tần suất ghi log (tính theo số bước G).")
    p.add_argument("--save_every", type=int, default=5000, help="Số bước lặp để lưu checkpoint mô hình.")
    p.add_argument("--sample_every", type=int, default=2500, help="Số bước lặp để sinh dải ảnh phản thực tế kiểm tra.")
    p.add_argument("--save_milestones", action="store_true", default=True, help="Lưu checkpoint theo từng mốc iters.")
    p.add_argument("--use_amp", action="store_true", default=False, help="Sử dụng Mixed Precision (AMP fp16) để tăng tốc.")
    p.add_argument("--resume", default=None, help="Đường dẫn checkpoint để train tiếp.")
    p.add_argument("--num_workers", type=int, default=4, help="Số worker nạp dữ liệu.")
    p.add_argument("--seed", type=int, default=0, help="Seed ngẫu nhiên.")
    return p.parse_args()


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_loss_curves(history_df, out_path):
    if len(history_df) < 2:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(history_df["iter"], history_df["loss_d"], color="tab:blue")
    axes[0].set_title("Discriminator Loss")
    axes[0].set_xlabel("Iterations")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history_df["iter"], history_df["loss_adv"], color="tab:orange")
    axes[1].set_title("Generator Adversarial Loss")
    axes[1].set_xlabel("Iterations")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(history_df["iter"], history_df["loss_kl"], color="tab:green")
    axes[2].set_title("Classifier Consistency Loss")
    axes[2].set_xlabel("Iterations")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    axes[3].plot(history_df["iter"], history_df["loss_rec"], color="tab:red", label="L1 Rec")
    axes[3].plot(history_df["iter"], history_df["loss_g"], color="tab:purple", label="Total G", alpha=0.7)
    axes[3].set_title("Reconstruction & Total G Loss")
    axes[3].set_xlabel("Iterations")
    axes[3].legend()
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    a = parse_args()
    seed_everything(a.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out_dir, exist_ok=True)

    log_file_path = os.path.join(a.out_dir, "train.log")
    log_file = open(log_file_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    # Lưu cấu hình thí nghiệm (config.json)
    config_path = os.path.join(a.out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(a), f, indent=2, ensure_ascii=False)
    log(f"Đã lưu cấu hình huấn luyện vào: {config_path}")

    # Dữ liệu: cân bằng theo bin điểm số
    df = balance_by_bin(pd.read_csv(a.csv_path), a.per_bin, a.seed)
    counts = pd.Series(score_to_bin_np(df["score"].values)).value_counts().sort_index()
    log(f"Số ảnh mỗi bin: {counts.to_dict()}")
    if len(counts) < N_BINS:
        log("CẢNH BÁO: có bin rỗng -> cGAN sẽ không học được điều kiện ở bin đó.")

    loader = DataLoader(
        CXRDataset(df, "score"),
        batch_size=a.batch_size,
        sampler=make_balanced_sampler(df),
        num_workers=a.num_workers,
        drop_last=True,
        pin_memory=True
    )
    batches = infinite(loader)

    # Mô hình
    f = load_target_classifier(a.classifier_path, a.target)
    X, D = Explainer().to(DEVICE), Discriminator().to(DEVICE)
    opt_g = torch.optim.Adam(X.parameters(), lr=a.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.0, 0.9))

    # Hỗ trợ AMP (Mixed Precision) nếu bật
    scaler_d = torch.amp.GradScaler(DEVICE.type, enabled=a.use_amp)
    scaler_g = torch.amp.GradScaler(DEVICE.type, enabled=a.use_amp)

    history = []
    history_csv_path = os.path.join(a.out_dir, "history.csv")
    start = 0

    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        X.load_state_dict(ck["X"])
        D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        if a.use_amp and "scaler_d" in ck:
            scaler_d.load_state_dict(ck["scaler_d"])
            scaler_g.load_state_dict(ck["scaler_g"])
        start = ck["it"]
        log(f"Resume thành công từ checkpoint: {a.resume} (iter {start})")
        if os.path.exists(history_csv_path):
            history = pd.read_csv(history_csv_path).to_dict(orient="records")

    def save_checkpoint(path, it):
        payload = {
            "X": X.state_dict(),
            "D": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "it": it,
            "args": vars(a)
        }
        if a.use_amp:
            payload["scaler_d"] = scaler_d.state_dict()
            payload["scaler_g"] = scaler_g.state_dict()
        atomic_save(payload, path)

    fixed_x01 = next(batches)[0][:4].to(DEVICE)  # 4 ảnh cố định để theo dõi tiến triển trực quan

    t_start = time.time()
    last_d_loss = 0.0

    log(f"Bắt đầu huấn luyện cGAN từ iter {start + 1} đến {a.iters} | AMP: {a.use_amp} | Device: {DEVICE}")

    for it in range(start + 1, a.iters + 1):
        x01, fx = next(batches)
        x01, fx = x01.to(DEVICE, non_blocking=True), fx.to(DEVICE, non_blocking=True)
        x = x01 * 2 - 1  # Đưa ảnh về miền [-1, 1] chuẩn của GAN
        B = x.size(0)
        bin_x = to_bin(fx)                           # Condition thật của x
        bin_c = to_bin(torch.rand(B, device=DEVICE))  # Condition ngẫu nhiên mục tiêu
        c_target = bin_center(bin_c)


        # Bước cập nhật Discriminator (D)
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=a.use_amp):
                x_fake = X(x, bin_c)

        opt_d.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=a.use_amp):
            pred_real = D(x, bin_x)
            pred_fake = D(x_fake, bin_c)
            l_d = d_loss(pred_real, pred_fake)

        scaler_d.scale(l_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()
        last_d_loss = l_d.item()


        # Bước cập nhật Generator & Encoder (G + E)
        if it % a.n_critic == 0:
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=a.use_amp):
                z = X.E(x)
                x_c = X.G(z, bin_c)  # Ảnh phản thực tế mục tiêu c

                # (A) Data consistency (GAN adversarial loss)
                l_adv = g_adv_loss(D(x_c, bin_c))

                # (B) Classifier consistency (Bernoulli KL divergence)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=False):
                    f_pred = f(((x_c.float() + 1) / 2))
                    l_f = classifier_consistency_loss(f_pred, c_target.float())

                # (C) Self-consistency & Cyclic reconstruction (L1 loss)
                x_self = X.G(z, bin_x)
                x_cyc = X.G(X.E(x_c), bin_x)
                l_rec = reconstruction_loss_l1(x, x_self, x_cyc)

                # Tổng hợp hàm mất mát của Generator
                l_g = a.lam_gan * l_adv + a.lam_f * l_f + a.lam_rec * l_rec

            scaler_g.scale(l_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            # Ghi log định kỳ
            if (it // a.n_critic) % a.log_every == 0:
                elapsed = time.time() - t_start
                it_done = it - start
                it_remain = a.iters - it
                speed = it_done / elapsed if elapsed > 0 else 0.0
                eta_sec = it_remain / speed if speed > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_sec)))

                row = {
                    "iter": it,
                    "loss_d": round(last_d_loss, 4),
                    "loss_adv": round(l_adv.item(), 4),
                    "loss_kl": round(l_f.item(), 4),
                    "loss_rec": round(l_rec.item(), 4),
                    "loss_g": round(l_g.item(), 4),
                    "elapsed_sec": round(elapsed, 1)
                }
                history.append(row)

                # Cập nhật history.csv
                pd.DataFrame(history).to_csv(history_csv_path, index=False)

                msg = (f"iter {it:05d}/{a.iters} | "
                       f"D: {row['loss_d']:.4f} | "
                       f"adv: {row['loss_adv']:.4f} | "
                       f"kl: {row['loss_kl']:.4f} | "
                       f"rec: {row['loss_rec']:.4f} | "
                       f"total_g: {row['loss_g']:.4f} | "
                       f"tốc độ: {speed:.1f} it/s | ETA: {eta_str}")
                log(msg)


        # Sinh dải ảnh phản thực tế trực quan
        if it % a.sample_every == 0:
            X.eval()
            with torch.no_grad():
                xcs, preds, fxs = progressive_counterfactuals(X, f, fixed_x01)
                for i in range(xcs.size(0)):
                    save_path = os.path.join(a.out_dir, f"sample_it{it:05d}_{i}.png")
                    save_strip_png(save_path, fixed_x01[i], xcs[i], preds[i], fxs[i].item())
            X.train()
            log(f"  -> Đã lưu ảnh minh họa tại iter {it} vào {a.out_dir}")

            # Cập nhật biểu đồ loss
            if len(history) > 1:
                save_loss_curves(pd.DataFrame(history), os.path.join(a.out_dir, "loss_curves.png"))

        # Lưu checkpoint mô hình
        if it % a.save_every == 0 or it == a.iters:
            # Luôn cập nhật checkpoint mới nhất
            save_checkpoint(os.path.join(a.out_dir, "explainer_last.pt"), it)
            # Lưu thêm checkpoint mốc nếu bật save_milestones (tránh rủi ro mode collapse)
            if a.save_milestones and it % a.save_every == 0 and it < a.iters:
                save_checkpoint(os.path.join(a.out_dir, f"explainer_it{it:05d}.pt"), it)
            log(f"Đã lưu checkpoint tại iter {it}")

    # Vẽ biểu đồ loss lần cuối khi hoàn thành
    if len(history) > 1:
        save_loss_curves(pd.DataFrame(history), os.path.join(a.out_dir, "loss_curves.png"))

    log(f"Huấn luyện hoàn tất! Checkpoint cuối cùng: {os.path.join(a.out_dir, 'explainer_last.pt')}")
    log_file.close()


if __name__ == "__main__":
    main()
