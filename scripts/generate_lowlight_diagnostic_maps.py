import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


SCENES = ["car-tsl", "car-toylta", "car-manycar"]
METHODS = [
    ("ours", "Ours"),
    ("2dgs", "2DGS"),
    ("3d_gabor", "3D-Gabor"),
]


def imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def resize_to(img: np.ndarray, shape_hw):
    h, w = shape_hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def abs_error_map(render: np.ndarray, gt: np.ndarray, vmax: float):
    err = np.mean(np.abs(render.astype(np.float32) / 255.0 - gt.astype(np.float32) / 255.0), axis=2)
    norm = np.clip(err / max(vmax, 1e-6), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), err


def sobel_edge_map(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    p99 = np.percentile(mag, 99.0)
    norm = np.clip(mag / max(p99, 1e-6), 0.0, 1.0)
    edge = (norm * 255).astype(np.uint8)
    edge_rgb = np.repeat(edge[:, :, None], 3, axis=2)
    return edge_rgb, mag, sx, sy


def _robust_signed_norm(x: np.ndarray, percentile: float = 99.2):
    scale = np.percentile(np.abs(x), percentile)
    return np.clip(x / max(scale, 1e-6), -1.0, 1.0)


def pseudo_feature_map(render: np.ndarray):
    gray_u8 = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray_u8).astype(np.float32) / 255.0

    # Three oriented Gabor response groups produce a fine pseudo-feature map:
    # color encodes local stripe/edge direction, and high-pass detail keeps
    # subtle pixel-level texture visible in zoomed panels.
    orientations = [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0]
    channels = []
    for theta in orientations:
        response = np.zeros_like(gray, dtype=np.float32)
        for lambd, sigma, gamma in [(4.0, 1.4, 0.55), (7.0, 2.1, 0.60), (11.0, 3.0, 0.65)]:
            kernel = cv2.getGaborKernel((15, 15), sigma, theta, lambd, gamma, 0.0, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            kernel /= np.sum(np.abs(kernel)) + 1e-6
            response += cv2.filter2D(gray, cv2.CV_32F, kernel)
        channels.append(_robust_signed_norm(response))

    smooth_small = cv2.GaussianBlur(gray, (0, 0), 0.9)
    smooth_large = cv2.GaussianBlur(gray, (0, 0), 3.0)
    high = _robust_signed_norm((gray - smooth_small) + 0.7 * (smooth_small - smooth_large), 98.8)

    response_rgb = np.stack(channels, axis=-1)
    energy = np.max(np.abs(response_rgb), axis=2)
    energy = np.clip(energy / max(np.percentile(energy, 99.0), 1e-6), 0.0, 1.0)
    chroma = response_rgb - np.mean(response_rgb, axis=2, keepdims=True)
    rgb = 0.54 + 0.70 * chroma + 0.20 * high[..., None] + 0.12 * energy[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)

    # Local contrast normalization suppresses broad smooth regions and makes
    # the map read like a dense feature response rather than a heatmap.
    mean = cv2.GaussianBlur(rgb, (0, 0), 1.4)
    sq_mean = cv2.GaussianBlur(rgb * rgb, (0, 0), 1.4)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 1e-6))
    rgb = np.clip(0.5 + 0.43 * (rgb - mean) / (std + 0.06), 0.0, 1.0)

    color_axis = rgb - np.mean(rgb, axis=2, keepdims=True)
    rgb = np.clip(0.52 + 1.55 * color_axis + 0.20 * (energy[..., None] - 0.5), 0.0, 1.0)

    pseudo = (rgb * 255.0).astype(np.uint8)
    pseudo = cv2.bilateralFilter(pseudo, d=3, sigmaColor=18, sigmaSpace=2)
    return pseudo


def choose_crop_box(gt: np.ndarray, crop_ratio: float):
    gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    score = cv2.GaussianBlur(np.sqrt(sx * sx + sy * sy), (0, 0), 3.0)
    h, w = gray.shape
    cw = max(32, int(w * crop_ratio))
    ch = max(32, int(h * crop_ratio))
    integral = cv2.integral(score)
    best = (-1.0, 0, 0)
    step_x = max(8, cw // 8)
    step_y = max(8, ch // 8)
    for y in range(0, max(1, h - ch), step_y):
        for x in range(0, max(1, w - cw), step_x):
            s = integral[y + ch, x + cw] - integral[y, x + cw] - integral[y + ch, x] + integral[y, x]
            # Avoid selecting only the border/background.
            center_bias = 1.0 - 0.25 * (abs((x + cw / 2) / w - 0.5) + abs((y + ch / 2) / h - 0.5))
            val = float(s) * center_bias
            if val > best[0]:
                best = (val, x, y)
    _, x, y = best
    return x, y, min(w, x + cw), min(h, y + ch)


def draw_label(draw, xy, text, font):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 5
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=(20, 20, 20))


def make_panel(gt, render, err_heat, edge, pseudo, crop_box, title, out_path):
    h, w = gt.shape[:2]
    panel_w = 420
    full_h = int(panel_w * h / w)
    gap = 8
    label_h = 28
    red = (190, 32, 32)
    font = ImageFont.load_default()

    def pil_resize(img, size):
        return Image.fromarray(img).resize(size, Image.Resampling.LANCZOS)

    gt_full = pil_resize(gt, (panel_w, full_h))
    render_full = pil_resize(render, (panel_w, full_h))
    err_full = pil_resize(err_heat, (panel_w, full_h))

    scale_x = panel_w / w
    scale_y = full_h / h
    x1, y1, x2, y2 = crop_box
    box = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
    for img in (gt_full, render_full, err_full):
        d = ImageDraw.Draw(img)
        d.rectangle(box, outline=red, width=3)

    crop_size = (panel_w, panel_w)
    gt_crop = pil_resize(gt[y1:y2, x1:x2], crop_size)
    render_crop = pil_resize(render[y1:y2, x1:x2], crop_size)
    edge_crop = pil_resize(edge[y1:y2, x1:x2], crop_size)
    pseudo_crop = pil_resize(pseudo[y1:y2, x1:x2], crop_size)
    err_crop = pil_resize(err_heat[y1:y2, x1:x2], crop_size)

    cols = 3
    rows = 3
    canvas_w = cols * panel_w + (cols - 1) * gap
    canvas_h = rows * panel_w + full_h - panel_w + rows * label_h + (rows - 1) * gap + 34
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 4), title, font=font, fill=(20, 20, 20))

    y = 30
    top_items = [("GT", gt_full), ("Render", render_full), ("Error map", err_full)]
    for i, (label, img) in enumerate(top_items):
        x = i * (panel_w + gap)
        draw_label(draw, (x + 6, y + 6), label, font)
        canvas.paste(img, (x, y + label_h))

    y += label_h + full_h + gap
    bottom_items = [
        ("GT crop", gt_crop),
        ("Render crop", render_crop),
        ("Error crop", err_crop),
        ("Edge crop", edge_crop),
        ("Pseudo-feature crop", pseudo_crop),
    ]
    positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    for (label, img), (cx, cy) in zip(bottom_items, positions):
        x = cx * (panel_w + gap)
        yy = y + cy * (panel_w + label_h + gap)
        draw_label(draw, (x + 6, yy + 6), label, font)
        canvas.paste(img, (x, yy + label_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def list_image_pairs(render_dir: Path, gt_dir: Path, max_images):
    renders = sorted([p for p in render_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    pairs = []
    for r in renders:
        g = gt_dir / r.name
        if not g.exists():
            # Some renderers use sequential render names but GT names can match.
            g = gt_dir / f"{r.stem}.png"
        if g.exists():
            pairs.append((r, g))
    return pairs[:max_images] if max_images else pairs


def method_model_path(scene, method, ours_root, baseline_root):
    if method == "ours":
        return ours_root / scene / "ours_best_d020_b10_t00015"
    if method == "2dgs":
        return baseline_root / scene / "01_2dgs"
    if method == "3d_gabor":
        return baseline_root / scene / "02_3d_gabor"
    raise ValueError(method)


def process_model(scene, method_key, method_label, model_path, args):
    split_root = model_path / args.split / f"ours_{args.iteration}"
    render_dir = split_root / "renders"
    gt_dir = split_root / "gt"
    if not render_dir.exists() or not gt_dir.exists():
        print(f"[skip] missing render/gt: {model_path}")
        return 0

    out_root = split_root / "diagnostics"
    err_dir = out_root / "error_maps"
    edge_dir = out_root / "edge_maps"
    pseudo_dir = out_root / "pseudo_feature_maps"
    panel_dir = out_root / "diagnostic_panels"
    pairs = list_image_pairs(render_dir, gt_dir, args.max_images)

    for render_path, gt_path in tqdm(pairs, desc=f"{scene}/{method_key}/{args.split}"):
        render = imread_rgb(render_path)
        gt = resize_to(imread_rgb(gt_path), render.shape[:2])
        err_heat, _ = abs_error_map(render, gt, args.error_vmax)
        edge, _, _, _ = sobel_edge_map(render)
        pseudo = pseudo_feature_map(render)

        name = render_path.name
        imwrite_rgb(err_dir / name, err_heat)
        imwrite_rgb(edge_dir / name, edge)
        imwrite_rgb(pseudo_dir / name, pseudo)

        if args.panels:
            crop = choose_crop_box(gt, args.crop_ratio)
            title = f"{scene} | {method_label} | {args.split}/{name}"
            make_panel(gt, render, err_heat, edge, pseudo, crop, title, panel_dir / name)

    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Generate error, edge, and pseudo-feature maps from rendered low-light comparisons.")
    parser.add_argument("--ours_root", required=True, help="Root directory containing VCNO-3DGS low-light model outputs.")
    parser.add_argument("--baseline_root", required=True, help="Root directory containing baseline low-light model outputs.")
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument("--methods", nargs="*", default=[m[0] for m in METHODS], choices=[m[0] for m in METHODS])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--error_vmax", type=float, default=0.20)
    parser.add_argument("--crop_ratio", type=float, default=0.24)
    parser.add_argument("--panels", action="store_true")
    args = parser.parse_args()

    ours_root = Path(args.ours_root)
    baseline_root = Path(args.baseline_root)
    method_labels = dict(METHODS)
    total = 0
    for scene in args.scenes:
        for method in args.methods:
            model = method_model_path(scene, method, ours_root, baseline_root)
            total += process_model(scene, method, method_labels[method], model, args)
    print(f"[done] processed {total} rendered images")


if __name__ == "__main__":
    main()
