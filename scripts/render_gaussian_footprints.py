import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from plyfile import PlyData
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import getWorld2View2, fov2focal
from utils.sh_utils import C0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussians(model_path: Path, iteration: int):
    if iteration < 0:
        point_root = model_path / "point_cloud"
        iters = []
        if point_root.exists():
            for p in point_root.iterdir():
                if p.is_dir() and p.name.startswith("iteration_"):
                    try:
                        iters.append(int(p.name.split("_")[-1]))
                    except ValueError:
                        pass
        if not iters:
            raise FileNotFoundError(f"No point_cloud/iteration_* under {model_path}")
        iteration = max(iters)

    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    names = {p.name for p in v.properties}

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    opacity = sigmoid(np.asarray(v["opacity"], dtype=np.float32))

    scale_names = sorted(
        [name for name in names if name.startswith("scale_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    if scale_names:
        raw_scales = np.stack([np.asarray(v[name]) for name in scale_names], axis=1).astype(np.float32)
        scales = np.exp(raw_scales)
    else:
        scales = np.ones((xyz.shape[0], 1), dtype=np.float32) * 0.01
    radius_world = np.max(scales, axis=1)

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.stack([np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], axis=1)
        colors = np.clip(sh * C0 + 0.5, 0.0, 1.0)
    elif {"red", "green", "blue"}.issubset(names):
        colors = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], axis=1) / 255.0
    else:
        colors = np.repeat(opacity[:, None], 3, axis=1)

    return {
        "iteration": iteration,
        "xyz": xyz,
        "opacity": opacity,
        "radius_world": radius_world,
        "colors": colors,
    }


def project_points(xyz, cam_info):
    w2c = getWorld2View2(cam_info.R, cam_info.T)
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    cam = xyz_h @ w2c
    z = cam[:, 2]

    fx = fov2focal(cam_info.FovX, cam_info.width)
    fy = fov2focal(cam_info.FovY, cam_info.height)
    x = fx * (cam[:, 0] / np.maximum(z, 1e-6)) + cam_info.width * 0.5
    y = fy * (cam[:, 1] / np.maximum(z, 1e-6)) + cam_info.height * 0.5
    return x, y, z, fx, fy


def draw_gaussian_view(
    gaussians,
    cam_info,
    out_path: Path,
    background="black",
    max_points=35000,
    min_opacity=0.02,
    radius_scale=1.0,
    min_radius=1.0,
    max_radius=16.0,
    blur=0.0,
):
    width, height = int(cam_info.width), int(cam_info.height)
    bg = (0, 0, 0, 255) if background == "black" else (255, 255, 255, 255)
    canvas = Image.new("RGBA", (width, height), bg)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    x, y, z, fx, fy = project_points(gaussians["xyz"], cam_info)
    visible = (
        (z > 1e-4)
        & (x >= -max_radius)
        & (x <= width + max_radius)
        & (y >= -max_radius)
        & (y <= height + max_radius)
        & (gaussians["opacity"] >= min_opacity)
    )
    idx = np.flatnonzero(visible)
    if idx.size == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.convert("RGB").save(out_path)
        return 0

    depth_order = idx[np.argsort(z[idx])[::-1]]
    if depth_order.size > max_points:
        score = gaussians["opacity"][depth_order] * gaussians["radius_world"][depth_order]
        keep = np.argpartition(score, -max_points)[-max_points:]
        depth_order = depth_order[keep]
        depth_order = depth_order[np.argsort(z[depth_order])[::-1]]

    radii = radius_scale * 0.5 * (fx + fy) * gaussians["radius_world"][depth_order] / np.maximum(z[depth_order], 1e-6)
    radii = np.clip(radii, min_radius, max_radius)
    alphas = np.clip(35 + 220 * gaussians["opacity"][depth_order], 0, 255).astype(np.uint8)
    colors = np.clip(gaussians["colors"][depth_order] * 255, 0, 255).astype(np.uint8)

    for px, py, r, color, alpha in zip(x[depth_order], y[depth_order], radii, colors, alphas):
        if px < -r or px > width + r or py < -r or py > height + r:
            continue
        draw.ellipse(
            (float(px - r), float(py - r), float(px + r), float(py + r)),
            fill=(int(color[0]), int(color[1]), int(color[2]), int(alpha)),
            outline=(255, 255, 255, min(int(alpha) + 30, 255)),
        )

    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur))
    canvas.alpha_composite(layer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)
    return int(idx.size)


def main():
    parser = argparse.ArgumentParser(description="Render per-view Gaussian footprint maps.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--images", default="images")
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--background", choices=["black", "white"], default="black")
    parser.add_argument("--max_points", type=int, default=35000)
    parser.add_argument("--min_opacity", type=float, default=0.02)
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--min_radius", type=float, default=1.0)
    parser.add_argument("--max_radius", type=float, default=16.0)
    parser.add_argument("--blur", type=float, default=0.0)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    gaussians = load_gaussians(model_path, args.iteration)
    scene_info = readColmapSceneInfo(
        args.source_path,
        args.images,
        args.eval,
        down_sample=False,
        init_point_num=-1,
    )
    iter_name = f"ours_{gaussians['iteration']}"

    splits = []
    if not args.skip_train:
        splits.append(("train", scene_info.train_cameras))
    if not args.skip_test and scene_info.test_cameras:
        splits.append(("test", scene_info.test_cameras))

    for split, cameras in splits:
        out_dir = model_path / split / iter_name / "gaussians"
        for cam in tqdm(cameras, desc=f"{model_path.name}/{split}/gaussians"):
            out_path = out_dir / f"{cam.image_name}.png"
            draw_gaussian_view(
                gaussians,
                cam,
                out_path,
                background=args.background,
                max_points=args.max_points,
                min_opacity=args.min_opacity,
                radius_scale=args.radius_scale,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                blur=args.blur,
            )
        print(f"Saved {split} Gaussian footprint maps to {out_dir}")


if __name__ == "__main__":
    main()
