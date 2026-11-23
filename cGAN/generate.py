import argparse
import os

import pandas as pd
import torch

from classifier import load_target_classifier
from common import DEVICE, N_BINS, seed_everything
from data import load_image
from models import load_explainer
from visualize import progressive_counterfactuals, save_strip_png

GROUPS = {
    "neg": lambda s: s < 0.2,
    "mild": lambda s: (s >= 0.4) & (s <= 0.6),
    "pos": lambda s: s > 0.8,
    "any": lambda s: s >= 0.0,
}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True, help="File CSV có cột `score`.")
    p.add_argument("--classifier_path", required=True, help = "Đường dẫn file checkpoint mô hình phân loại.")
    p.add_argument("--explainer_path", required=True, help = "Đường dẫn file checkpoint cGAN.")
    p.add_argument("--target", default="covid19", help = "Tên nhãn mục tiêu cần lấy xác suất.")
    p.add_argument("--group", choices=list(GROUPS), default="any", help="Chọn ảnh theo phân loại từ classifier.")
    p.add_argument("--n", type=int, default=8, help = "Số lượng ảnh phản thực tạo ra.")
    p.add_argument("--out_dir", default="vis")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    seed_everything(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv_path)
    df = df[GROUPS[a.group](df["score"])]
    df = df.sample(min(a.n, len(df)), random_state=a.seed).reset_index(drop=True)

    f = load_target_classifier(a.classifier_path, a.target)
    X = load_explainer(a.explainer_path)

    rows = []
    for i, r in df.iterrows():
        x01 = load_image(r["path"])[None].to(DEVICE)
        xcs, preds, fx = progressive_counterfactuals(X, f, x01)
        out = os.path.join(a.out_dir, f"{a.target.replace(' ', '_')}_{i:03d}.png")
        save_strip_png(out, x01[0], xcs[0], preds[0], fx[0].item())
        rows.append({"path": r["path"], "out": out, "f_x": fx[0].item(),
                     **{f"f_xc_bin{b}": preds[0, b].item() for b in range(N_BINS)}})
        print("Đã lưu", out, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
