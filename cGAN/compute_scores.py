import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from classifier import load_target_classifier
from common import DEVICE
from data import CXRDataset


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Tính điểm dự đoán của mô hình phân loại cho từng ảnh và lưu vào CSV.")
    p.add_argument("--csv_path", required=True, help="Đường dẫn file CSV chứa đường dẫn ảnh X-quang.")
    p.add_argument("--classifier_path", required=True, help="Đường dẫn file checkpoint mô hình phân loại.")
    p.add_argument("--target", default="covid19", help="Tên nhãn mục tiêu cần lấy xác suất.")
    p.add_argument("--output_path", required=True, help="Đường dẫn file CSV đầu ra (có cột điểm cho từng ảnh).")
    p.add_argument("--batch_size", type=int, default=64, help="Kích thước batch.")
    p.add_argument("--num_workers", type=int, default=4, help="Số workers cho DataLoader.")
    a = p.parse_args()

    df = pd.read_csv(a.csv_path)
    f = load_target_classifier(a.classifier_path, a.target)
    dl = DataLoader(CXRDataset(df), batch_size=a.batch_size, num_workers=a.num_workers)
    scores = []
    for i, (x, _) in enumerate(dl):
        scores.append(f(x.to(DEVICE)).cpu())
        if i % 50 == 0:
            print(f"{i * a.batch_size}/{len(df)}", flush=True)
    df["score"] = torch.cat(scores).numpy()
    df.to_csv(a.output_path, index=False)
    print(f"Đã ghi {a.output_path}; phân bố bin:")
    bin_labels = [f"[{i/10:.1f}, {(i+1)/10:.1f})" for i in range(9)] + ["[0.9, 1.0]"]
    print(pd.cut(df["score"], bins=[i / 10 for i in range(11)], labels=bin_labels, include_lowest=True).value_counts().sort_index())


if __name__ == "__main__":
    main()
