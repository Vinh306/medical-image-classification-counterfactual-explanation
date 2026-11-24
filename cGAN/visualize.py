import numpy as np
import torch
from PIL import Image, ImageDraw

from common import DEVICE, N_BINS


@torch.no_grad()
def progressive_counterfactuals(explainer, f, x01):
    B = x01.size(0)
    z = explainer.E(x01 * 2 - 1)
    outs = []
    for b in range(N_BINS):
        y = torch.full((B,), b, dtype=torch.long, device=x01.device)
        outs.append((explainer.G(z, y) + 1) / 2)
    xcs = torch.stack(outs, dim=1)
    preds = torch.stack([f(o) for o in outs], dim=1)
    return xcs, preds, f(x01)


def difference_map(xcs):
    d = (xcs[:, 0] - xcs[:, -1]).abs()
    return d / d.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)


def _heat(d):
    import matplotlib
    cmap = matplotlib.colormaps["magma"]
    rgba = cmap(d.squeeze(0).cpu().numpy())
    return torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).float()


def save_strip_png(path: str, x01, xcs, preds, fx):
    gray = lambda t: t.detach().cpu().repeat(3, 1, 1)
    d = difference_map(xcs[None])[0]
    tiles = [gray(x01)] + [gray(t) for t in xcs] + [0.4 * gray(x01) + 0.6 * _heat(d)]
    names = [f"input f={fx:.2f}"] + [f"c={(b + 0.5) / N_BINS:.2f} f={p:.2f}" for b, p in enumerate(preds.tolist())]
    names.append("|x(c=0)-x(c=1)|")

    row = torch.cat(tiles, dim=2).clamp(0, 1)
    img = Image.fromarray((row.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    H, W = x01.shape[-2:]
    canvas = Image.new("RGB", (img.width, img.height + 18), (0, 0, 0))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, s in enumerate(names):
        draw.text((i * W + 4, H + 3), s, fill=(255, 255, 255))
    canvas.save(path, dpi=(300, 300))
