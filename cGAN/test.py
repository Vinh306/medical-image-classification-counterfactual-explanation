import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from classifier import load_target_classifier
from common import DEVICE, N_BINS, seed_everything
from data import CXRDataset
from metrics import build_fid_net, counterfactual_flip, fid_from_features, inception_features
from models import load_explainer


@torch.no_grad()
def run_group(df, target_bin, X, f, fid_net, bs, num_workers, tau):
    loader = DataLoader(CXRDataset(df), batch_size=bs, shuffle=False, num_workers=num_workers)
    real_feats, cf_feats, flips = [], [], []
    for x01, _ in loader:
        x01 = x01.to(DEVICE)
        y = torch.full((x01.size(0),), target_bin, dtype=torch.long, device=DEVICE)
        xc01 = (X(x01 * 2 - 1, y) + 1) / 2
        flips.append(counterfactual_flip(f(x01), f(xc01), tau).cpu())
        real_feats.append(inception_features(fid_net, x01).cpu())
        cf_feats.append(inception_features(fid_net, xc01).cpu())
    return (torch.cat(flips), torch.cat(real_feats).numpy().astype(np.float64),
            torch.cat(cf_feats).numpy().astype(np.float64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True, help="File CSV có cột `score`.")
    p.add_argument("--classifier_path", required=True, help = "Đường dẫn file checkpoint mô hình phân loại.")
    p.add_argument("--explainer_path", required=True, help = "Đường dẫn file checkpoint mô hình explainer.")
    p.add_argument("--target", default="covid19", help = "Tên nhãn mục tiêu cần lấy xác suất.")
    p.add_argument("--fid_net", default=None, help="Inception đã fine-tune; bỏ trống = ImageNet")
    p.add_argument("--max_images", type=int, default=3000, help="Số ảnh tối đa mỗi nhóm (âm tính và dương tính).")
    p.add_argument("--tau", type=float, default=0.8, help = "Ngưỡng xác định ảnh phản thực là thành công.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_json", default="results.json")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    seed_everything(a.seed)

    df = pd.read_csv(a.csv_path)
    neg, pos = df[df["score"] < 0.2], df[df["score"] > 0.8]
    neg = neg.sample(min(a.max_images, len(neg)), random_state=a.seed)
    pos = pos.sample(min(a.max_images, len(pos)), random_state=a.seed)
    print(f"Nhóm âm: {len(neg)} ảnh | nhóm dương: {len(pos)} ảnh", flush=True)
    for name, g in [("âm", neg), ("dương", pos)]:
        if len(g) < 2048:
            print(f"CẢNH BÁO: nhóm {name} chỉ có {len(g)} ảnh (<2048) nên FID kém tin cậy.")

    f = load_target_classifier(a.classifier_path, a.target)
    X = load_explainer(a.explainer_path)
    fid_net = build_fid_net(a.fid_net)

    flip_n, real_n, cf_n = run_group(neg, N_BINS - 1, X, f, fid_net, a.batch_size, a.num_workers, a.tau)   # neg -> pos (c=1)
    flip_p, real_p, cf_p = run_group(pos, 0, X, f, fid_net, a.batch_size, a.num_workers, a.tau)            # pos -> neg (c=0)
    real_all = np.concatenate([real_n, real_p], axis=0)
    cf_all = np.concatenate([cf_n, cf_p], axis=0)

    results = {
        "target": a.target,
        "n_total": len(real_all),
        "n_neg": len(neg), "n_pos": len(pos), "tau": a.tau,
        "CV": float(torch.cat([flip_n, flip_p]).float().mean()),
        "CV_neg_to_pos": float(flip_n.float().mean()),
        "CV_pos_to_neg": float(flip_p.float().mean()),
        "FID_total": fid_from_features(real_all, cf_all),
        "FID_neg_to_pos": fid_from_features(real_n, cf_n),
        "FID_pos_to_neg": fid_from_features(real_p, cf_p),
        "fid_net": a.fid_net or "ImageNet",
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    with open(a.out_json, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
