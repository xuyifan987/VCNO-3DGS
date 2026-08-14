import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import fov2focal, getWorld2View2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_rot(q):
    q = q.astype(np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def load_gaussians(model_path, iteration):
    point_root = model_path / "point_cloud"
    if iteration < 0:
        iters = []
        for p in point_root.glob("iteration_*"):
            try:
                iters.append(int(p.name.split("_")[-1]))
            except ValueError:
                pass
        if not iters:
            raise FileNotFoundError(f"No point_cloud/iteration_* under {model_path}")
        iteration = max(iters)

    ply_path = point_root / f"iteration_{iteration}" / "point_cloud.ply"
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    names = {p.name for p in v.properties}

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    opacity = sigmoid(np.asarray(v["opacity"], dtype=np.float32))

    scale_names = sorted([n for n in names if n.startswith("scale_")], key=lambda n: int(n.split("_")[-1]))
    scales = np.exp(np.stack([np.asarray(v[n]) for n in scale_names], axis=1).astype(np.float32))
    if scales.shape[1] == 1:
        scales = np.repeat(scales, 2, axis=1)

    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        quat = np.stack([np.asarray(v[f"rot_{i}"]) for i in range(4)], axis=1).astype(np.float32)
        rot = quat_to_rot(quat)
    else:
        rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], xyz.shape[0], axis=0)

    return {
        "iteration": iteration,
        "xyz": xyz,
        "opacity": opacity,
        "scale0": scales[:, 0],
        "scale1": scales[:, min(1, scales.shape[1] - 1)],
        "axis0": rot[:, :, 0],
        "axis1": rot[:, :, 1],
    }


def project_points(xyz, cam_info):
    # Match the renderer convention: Camera.world_view_transform stores
    # getWorld2View2(...).T and points are multiplied as row vectors.
    w2c = getWorld2View2(cam_info.R, cam_info.T).T
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    cam = xyz_h @ w2c
    z = cam[:, 2]
    fx = fov2focal(cam_info.FovX, cam_info.width)
    fy = fov2focal(cam_info.FovY, cam_info.height)
    x = fx * (cam[:, 0] / np.maximum(z, 1e-6)) + cam_info.width * 0.5
    y = fy * (cam[:, 1] / np.maximum(z, 1e-6)) + cam_info.height * 0.5
    return x, y, z


def sample_colors(render_image, x, y):
    arr = np.asarray(render_image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    xi = np.clip(np.rint(x).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(y).astype(np.int32), 0, h - 1)
    return arr[yi, xi]


def select_fair_indices(idx, p0x, p0y, ra, rb, opacity, width, height, args):
    if args.fair_points <= 0 or idx.size <= args.fair_points:
        return idx

    # Pick visible Gaussians in a screen-space stratified way so that one method
    # cannot look denser simply because many large projected ellipses overlap in
    # the same image region. This is for qualitative distribution visualization,
    # not for metric computation.
    gx = np.clip((p0x[idx] / max(width, 1) * args.fair_grid_x).astype(np.int32), 0, args.fair_grid_x - 1)
    gy = np.clip((p0y[idx] / max(height, 1) * args.fair_grid_y).astype(np.int32), 0, args.fair_grid_y - 1)
    cell = gy * args.fair_grid_x + gx
    score = opacity[idx] * np.sqrt(np.maximum(ra[idx] * rb[idx], 1e-6))

    order = np.argsort(score)[::-1]
    per_cell_quota = max(1, int(np.ceil(args.fair_points / float(args.fair_grid_x * args.fair_grid_y))))
    counts = {}
    selected = []
    deferred = []
    for local in order:
        c = int(cell[local])
        if counts.get(c, 0) < per_cell_quota:
            selected.append(idx[local])
            counts[c] = counts.get(c, 0) + 1
            if len(selected) >= args.fair_points:
                break
        else:
            deferred.append(idx[local])

    if len(selected) < args.fair_points and deferred:
        selected.extend(deferred[: args.fair_points - len(selected)])
    return np.asarray(selected[: args.fair_points], dtype=np.int64)


def alpha_over(dst_rgb, dst_alpha, src_rgb, src_alpha):
    one_minus = 1.0 - src_alpha
    out_alpha = src_alpha + dst_alpha * one_minus
    out_rgb = src_rgb * src_alpha[..., None] + dst_rgb * (dst_alpha * one_minus)[..., None]
    valid = out_alpha > 1e-6
    dst_rgb[valid] = out_rgb[valid] / out_alpha[valid, None]
    dst_alpha[valid] = out_alpha[valid]


def splat_shaded_ellipsoid(rgb, acc_alpha, cx, cy, a_vec, b_vec, color, opacity, args):
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))
    if args.force_min_radius:
        if a_len < args.min_radius:
            a_vec = np.array([args.min_radius, 0.0], dtype=np.float32) if a_len < 1e-6 else a_vec * (args.min_radius / a_len)
            a_len = args.min_radius
        if b_len < args.min_radius:
            b_vec = np.array([0.0, args.min_radius], dtype=np.float32) if b_len < 1e-6 else b_vec * (args.min_radius / b_len)
            b_len = args.min_radius
    elif a_len < args.min_radius or b_len < args.min_radius:
        return
    max_len = max(a_len, b_len)
    radius_cap = args.fair_max_radius if args.fair_mode else args.max_radius
    if max_len > radius_cap:
        shrink = radius_cap / max_len
        a_vec = a_vec * shrink
        b_vec = b_vec * shrink
        a_len *= shrink
        b_len *= shrink
    if a_len < args.min_radius or b_len < args.min_radius:
        return

    pts = np.array([a_vec + b_vec, a_vec - b_vec, -a_vec + b_vec, -a_vec - b_vec], dtype=np.float32)
    xmin = max(0, int(np.floor(cx + pts[:, 0].min())) - 1)
    xmax = min(rgb.shape[1], int(np.ceil(cx + pts[:, 0].max())) + 2)
    ymin = max(0, int(np.floor(cy + pts[:, 1].min())) - 1)
    ymax = min(rgb.shape[0], int(np.ceil(cy + pts[:, 1].max())) + 2)
    if xmax <= xmin or ymax <= ymin:
        return

    yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float32)
    d = np.stack([xx - cx, yy - cy], axis=-1)
    A = np.stack([a_vec, b_vec], axis=1).astype(np.float32)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-5:
        return
    inv_a = np.linalg.inv(A)
    uv = d @ inv_a.T
    r2 = uv[..., 0] ** 2 + uv[..., 1] ** 2
    mask = r2 <= 1.0
    if not np.any(mask):
        return

    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    light = np.array(args.light_dir, dtype=np.float32)
    light = light / max(float(np.linalg.norm(light)), 1e-6)
    normal_dot = np.clip(uv[..., 0] * light[0] + uv[..., 1] * light[1] + nz * light[2], 0.0, 1.0)
    shade = args.ambient + args.diffuse * normal_dot
    rim = np.clip(1.0 - nz, 0.0, 1.0) ** args.rim_power
    shade = np.clip(shade + args.rim_strength * rim, 0.0, 1.35)

    src_rgb = np.clip((color[None, None, :] / 255.0) * shade[..., None], 0.0, 1.0)
    soft_edge = np.clip((1.0 - r2) / max(args.edge_softness, 1e-4), 0.0, 1.0)
    if args.fair_mode:
        src_alpha = np.clip(args.fair_alpha * (soft_edge ** 0.5), 0.0, args.fair_alpha)
    else:
        src_alpha = np.clip(opacity * args.opacity_scale * (soft_edge ** 0.5), 0.0, args.max_alpha)
    src_alpha = np.where(mask, src_alpha, 0.0).astype(np.float32)

    dst_rgb = rgb[ymin:ymax, xmin:xmax]
    dst_alpha = acc_alpha[ymin:ymax, xmin:xmax]
    alpha_over(dst_rgb, dst_alpha, src_rgb.astype(np.float32), src_alpha)


def render_ellipsoids(gaussians, cam_info, render_path, out_path, args):
    width, height = int(cam_info.width), int(cam_info.height)
    render_img = Image.open(render_path).convert("RGB").resize((width, height))
    base = Image.blend(Image.new("RGB", (width, height), tuple(args.background_color)), render_img, args.background_strength)
    rgb = np.asarray(base, dtype=np.float32) / 255.0
    acc_alpha = np.zeros((height, width), dtype=np.float32)

    xyz = gaussians["xyz"]
    p0x, p0y, z = project_points(xyz, cam_info)
    p1x, p1y, _ = project_points(xyz + gaussians["axis0"] * gaussians["scale0"][:, None], cam_info)
    p2x, p2y, _ = project_points(xyz + gaussians["axis1"] * gaussians["scale1"][:, None], cam_info)

    a = np.stack([p1x - p0x, p1y - p0y], axis=1) * args.radius_scale
    b = np.stack([p2x - p0x, p2y - p0y], axis=1) * args.radius_scale
    ra = np.linalg.norm(a, axis=1)
    rb = np.linalg.norm(b, axis=1)
    visible = (
        (z > 1e-4)
        & (p0x >= -args.max_radius)
        & (p0x <= width + args.max_radius)
        & (p0y >= -args.max_radius)
        & (p0y <= height + args.max_radius)
    )
    if not args.no_filter:
        visible = visible & (gaussians["opacity"] >= args.min_opacity) & (np.maximum(ra, rb) >= args.min_radius)
    idx = np.flatnonzero(visible)
    if args.fair_mode:
        idx = select_fair_indices(idx, p0x, p0y, ra, rb, gaussians["opacity"], width, height, args)
    if args.max_points > 0 and idx.size > args.max_points:
        score = gaussians["opacity"][idx] * np.maximum(ra[idx], rb[idx])
        idx = idx[np.argpartition(score, -args.max_points)[-args.max_points:]]

    idx = idx[np.argsort(z[idx])[::-1]]
    colors = sample_colors(render_img, p0x[idx], p0y[idx]).astype(np.float32)
    if args.color_lift > 0:
        colors = colors * (1.0 - args.color_lift) + 255.0 * args.color_lift
    if args.fair_mode and args.fair_gray > 0:
        gray = np.mean(colors, axis=1, keepdims=True)
        colors = colors * (1.0 - args.fair_gray) + gray * args.fair_gray
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    for j, color in zip(idx, colors):
        splat_shaded_ellipsoid(
            rgb,
            acc_alpha,
            float(p0x[j]),
            float(p0y[j]),
            a[j],
            b[j],
            color,
            args.fair_alpha if args.fair_mode else max(float(gaussians["opacity"][j]), args.opacity_floor),
            args,
        )

    top = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), "RGB")
    if args.panel:
        gap = args.panel_gap
        panel = Image.new("RGB", (width, height * 2 + gap), (255, 255, 255))
        panel.paste(top, (0, 0))
        panel.paste(render_img, (0, height + gap))
        out = panel
    else:
        out = top
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def overview_axes(args):
    az = np.deg2rad(args.overview_azim)
    el = np.deg2rad(args.overview_elev)
    forward = np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)], dtype=np.float32)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-6)
    return right, up, forward


def render_overview_ellipsoids(gaussians, render_path, out_path, args):
    width, height = args.overview_width, args.overview_height
    base = Image.new("RGB", (width, height), tuple(args.background_color))
    rgb = np.asarray(base, dtype=np.float32) / 255.0
    acc_alpha = np.zeros((height, width), dtype=np.float32)

    xyz = gaussians["xyz"]
    center = np.median(xyz, axis=0)
    centered = xyz - center[None, :]
    right, up, forward = overview_axes(args)
    vx = centered @ right
    vy = centered @ up
    vz = centered @ forward

    if args.overview_fit_percentile >= 100:
        xmin, xmax = float(vx.min()), float(vx.max())
        ymin, ymax = float(vy.min()), float(vy.max())
    else:
        lo = (100.0 - args.overview_fit_percentile) * 0.5
        hi = 100.0 - lo
        xmin, xmax = np.percentile(vx, [lo, hi])
        ymin, ymax = np.percentile(vy, [lo, hi])

    span_x = max(float(xmax - xmin), 1e-6)
    span_y = max(float(ymax - ymin), 1e-6)
    scale = min(width / (span_x * args.overview_margin), height / (span_y * args.overview_margin))
    cx0 = (xmin + xmax) * 0.5
    cy0 = (ymin + ymax) * 0.5
    p0x = width * 0.5 + (vx - cx0) * scale
    p0y = height * 0.5 - (vy - cy0) * scale

    a = np.stack(
        [gaussians["axis0"] @ right, -(gaussians["axis0"] @ up)],
        axis=1,
    ) * (gaussians["scale0"][:, None] * scale * args.radius_scale)
    b = np.stack(
        [gaussians["axis1"] @ right, -(gaussians["axis1"] @ up)],
        axis=1,
    ) * (gaussians["scale1"][:, None] * scale * args.radius_scale)

    idx = np.arange(xyz.shape[0])
    if args.min_opacity > 0:
        idx = idx[gaussians["opacity"][idx] >= args.min_opacity]
    if args.max_points > 0 and idx.size > args.max_points:
        score = gaussians["opacity"][idx] * np.maximum(np.linalg.norm(a[idx], axis=1), np.linalg.norm(b[idx], axis=1))
        idx = idx[np.argpartition(score, -args.max_points)[-args.max_points:]]
    idx = idx[np.argsort(vz[idx])]

    colors = np.full((idx.size, 3), args.overview_color, dtype=np.float32)
    if render_path and Path(render_path).exists():
        # Optional tint from the nearest projected pixel of a reference render.
        ref = Image.open(render_path).convert("RGB").resize((width, height))
        sampled = sample_colors(ref, np.clip(p0x[idx], 0, width - 1), np.clip(p0y[idx], 0, height - 1)).astype(np.float32)
        colors = colors * (1.0 - args.overview_color_mix) + sampled * args.overview_color_mix
    if args.color_lift > 0:
        colors = colors * (1.0 - args.color_lift) + 255.0 * args.color_lift
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    for j, color in zip(idx, colors):
        splat_shaded_ellipsoid(
            rgb,
            acc_alpha,
            float(p0x[j]),
            float(p0y[j]),
            a[j],
            b[j],
            color,
            max(float(gaussians["opacity"][j]), args.opacity_floor),
            args,
        )

    top = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), "RGB")
    if args.panel and render_path and Path(render_path).exists():
        ref = Image.open(render_path).convert("RGB").resize((width, height))
        panel = Image.new("RGB", (width, height * 2 + args.panel_gap), (255, 255, 255))
        panel.paste(top, (0, 0))
        panel.paste(ref, (0, height + args.panel_gap))
        out = panel
    else:
        out = top
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def find_render_path(model_path, split, iteration, image_name, view_index=None):
    render_dir = model_path / split / f"ours_{iteration}" / "renders"
    for suffix in (".png", ".jpg", ".jpeg"):
        p = render_dir / f"{image_name}{suffix}"
        if p.exists():
            return p
    if view_index is not None:
        for suffix in (".png", ".jpg", ".jpeg"):
            p = render_dir / f"{view_index:05d}{suffix}"
            if p.exists():
                return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Render view-aligned 3D-looking Gaussian ellipsoid panels.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--images", default="images")
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--max_points", type=int, default=30000)
    parser.add_argument("--min_opacity", type=float, default=0.025)
    parser.add_argument("--radius_scale", type=float, default=1.7)
    parser.add_argument("--min_radius", type=float, default=0.8)
    parser.add_argument("--max_radius", type=float, default=28.0)
    parser.add_argument("--background_strength", type=float, default=0.55)
    parser.add_argument("--background_color", type=int, nargs=3, default=[20, 20, 20])
    parser.add_argument("--opacity_scale", type=float, default=0.82)
    parser.add_argument("--opacity_floor", type=float, default=0.0)
    parser.add_argument("--max_alpha", type=float, default=0.82)
    parser.add_argument("--edge_softness", type=float, default=0.18)
    parser.add_argument("--ambient", type=float, default=0.45)
    parser.add_argument("--diffuse", type=float, default=0.72)
    parser.add_argument("--rim_strength", type=float, default=0.18)
    parser.add_argument("--rim_power", type=float, default=2.0)
    parser.add_argument("--light_dir", type=float, nargs=3, default=[-0.45, -0.65, 1.0])
    parser.add_argument("--color_lift", type=float, default=0.0)
    parser.add_argument("--fair_mode", action="store_true", help="Use equal visual budget, fixed alpha, and capped radii for fair qualitative comparison.")
    parser.add_argument("--fair_points", type=int, default=12000, help="Number of visible Gaussians to draw in fair_mode.")
    parser.add_argument("--fair_alpha", type=float, default=0.34, help="Fixed per-Gaussian alpha used in fair_mode.")
    parser.add_argument("--fair_max_radius", type=float, default=9.0, help="Projected radius cap in fair_mode.")
    parser.add_argument("--fair_grid_x", type=int, default=40, help="Screen-space stratification grid width in fair_mode.")
    parser.add_argument("--fair_grid_y", type=int, default=24, help="Screen-space stratification grid height in fair_mode.")
    parser.add_argument("--fair_gray", type=float, default=0.35, help="Desaturate sampled colors in fair_mode to reduce visual clutter.")
    parser.add_argument("--panel", action="store_true", help="Stack Gaussian ellipsoid view above the rendered image.")
    parser.add_argument("--panel_gap", type=int, default=6)
    parser.add_argument("--no_filter", action="store_true", help="Draw all projected Gaussians in the selected view; disables opacity/radius/count filtering.")
    parser.add_argument("--force_min_radius", action="store_true", help="Keep very small projected Gaussians visible by enlarging them to min_radius.")
    parser.add_argument("--view_index", type=int, default=None, help="Render only one camera index within each selected split.")
    parser.add_argument("--overview", action="store_true", help="Render all Gaussians once from a fixed orthographic 3D overview instead of a dataset camera frustum.")
    parser.add_argument("--overview_width", type=int, default=1178)
    parser.add_argument("--overview_height", type=int, default=660)
    parser.add_argument("--overview_azim", type=float, default=22.0)
    parser.add_argument("--overview_elev", type=float, default=12.0)
    parser.add_argument("--overview_margin", type=float, default=1.12)
    parser.add_argument("--overview_fit_percentile", type=float, default=100.0)
    parser.add_argument("--overview_color", type=int, nargs=3, default=[170, 174, 164])
    parser.add_argument("--overview_color_mix", type=float, default=0.10)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    gaussians = load_gaussians(model_path, args.iteration)
    iteration = gaussians["iteration"]

    if args.overview:
        render_path = None
        if args.view_index is not None:
            scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval, down_sample=False, init_point_num=-1)
            cameras = scene_info.test_cameras or scene_info.train_cameras
            if 0 <= args.view_index < len(cameras):
                render_path = find_render_path(model_path, "test" if scene_info.test_cameras else "train", iteration, cameras[args.view_index].image_name, args.view_index)
        out_dir = model_path / "overview" / f"ours_{iteration}" / "gaussian_3d_overview_panels" if args.panel else model_path / "overview" / f"ours_{iteration}" / "gaussian_3d_overview"
        render_overview_ellipsoids(gaussians, render_path, out_dir / "all_gaussians.png", args)
        print(f"Saved overview output to {out_dir}")
        return

    scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval, down_sample=False, init_point_num=-1)

    splits = []
    if not args.skip_train:
        splits.append(("train", scene_info.train_cameras))
    if not args.skip_test and scene_info.test_cameras:
        splits.append(("test", scene_info.test_cameras))

    for split, cameras in splits:
        if args.fair_mode:
            out_name = "gaussian_3d_fair_panels" if args.panel else "gaussian_3d_fair"
        else:
            out_name = "gaussian_3d_all_panels" if args.no_filter and args.panel else "gaussian_3d_all" if args.no_filter else "gaussian_3d_panels" if args.panel else "gaussian_3d"
        out_dir = model_path / split / f"ours_{iteration}" / out_name
        iterable = list(enumerate(cameras))
        if args.view_index is not None:
            iterable = [(i, c) for i, c in iterable if i == args.view_index]
        for view_index, cam in tqdm(iterable, desc=f"{model_path.name}/{split}/ellipsoids"):
            render_path = find_render_path(model_path, split, iteration, cam.image_name, view_index)
            if render_path is None:
                continue
            render_ellipsoids(gaussians, cam, render_path, out_dir / f"{cam.image_name}.png", args)
        print(f"Saved {split} outputs to {out_dir}")


if __name__ == "__main__":
    main()
