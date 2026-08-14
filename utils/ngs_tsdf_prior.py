import math
import numpy as np
import torch
from tqdm import tqdm


def rigid_transform(xyz, transform):
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=xyz.dtype)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h, im_w = depth_im.shape
    max_depth = np.max(depth_im)
    if max_depth <= 0:
        max_depth = 1.0
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
        * np.array([0, max_depth, max_depth, max_depth, max_depth])
        / cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
        * np.array([0, max_depth, max_depth, max_depth, max_depth])
        / cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth]),
    ])
    return rigid_transform(view_frust_pts.T, cam_pose).T


class TorchTSDFVolume:
    """Torch implementation of the TSDF update used by GSPrior's fusion code."""

    def __init__(
        self,
        vol_bnds,
        voxel_size,
        sdf_trunc,
        device="cuda",
        chunk_size=262144,
        max_voxels=2_000_000,
    ):
        vol_bnds = np.asarray(vol_bnds, dtype=np.float32)
        assert vol_bnds.shape == (3, 2), "vol_bnds should have shape (3, 2)."

        self.device = torch.device(device)
        self._vol_bnds = vol_bnds.copy()
        self._voxel_size = float(voxel_size)
        self._trunc_margin = float(sdf_trunc)
        self.chunk_size = int(chunk_size)
        self.max_voxels = int(max_voxels)

        self._fit_volume_size()
        self._vol_origin_np = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)
        self._vol_origin = torch.as_tensor(self._vol_origin_np, dtype=torch.float32, device=self.device)

        print(
            "TSDF voxel volume: {} x {} x {} - # voxels: {:,}, voxel_size: {:.5f}".format(
                self._vol_dim[0],
                self._vol_dim[1],
                self._vol_dim[2],
                int(np.prod(self._vol_dim)),
                self._voxel_size,
            )
        )

        self._tsdf_vol = torch.ones(tuple(self._vol_dim.tolist()), dtype=torch.float32, device=self.device)
        self._weight_vol = torch.zeros_like(self._tsdf_vol)

    def _fit_volume_size(self):
        extent = np.maximum(self._vol_bnds[:, 1] - self._vol_bnds[:, 0], 1e-6)
        vol_dim = np.ceil(extent / self._voxel_size).astype(np.int64)
        voxels = int(np.prod(vol_dim))
        if voxels > self.max_voxels:
            scale = (voxels / float(self.max_voxels)) ** (1.0 / 3.0)
            self._voxel_size *= scale
            vol_dim = np.ceil(extent / self._voxel_size).astype(np.int64)
        self._vol_dim = np.maximum(vol_dim, 2)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def vol_origin(self):
        return self._vol_origin_np

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.0):
        depth = torch.as_tensor(depth_im, dtype=torch.float32, device=self.device)
        cam_intr = torch.as_tensor(cam_intr, dtype=torch.float32, device=self.device)
        cam_pose = torch.as_tensor(cam_pose, dtype=torch.float32, device=self.device)
        world_to_cam = torch.linalg.inv(cam_pose)

        im_h, im_w = depth.shape
        nx, ny, nz = [int(v) for v in self._vol_dim]
        yz = ny * nz
        total = nx * ny * nz

        for start in range(0, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            idx = torch.arange(start, end, device=self.device, dtype=torch.long)
            z = idx % nz
            y = (idx // nz) % ny
            x = idx // yz

            pts = torch.stack((x, y, z), dim=1).float()
            pts = pts * self._voxel_size + self._vol_origin
            pts_h = torch.cat((pts, torch.ones((pts.shape[0], 1), device=self.device)), dim=1)
            cam_pts = pts_h @ world_to_cam.T
            cam_z = cam_pts[:, 2]
            safe_z = cam_z.clamp_min(1e-6)

            pix_x = torch.round(cam_intr[0, 0] * (cam_pts[:, 0] / safe_z) + cam_intr[0, 2]).long()
            pix_y = torch.round(cam_intr[1, 1] * (cam_pts[:, 1] / safe_z) + cam_intr[1, 2]).long()

            valid = (
                (cam_z > 0)
                & (pix_x >= 0)
                & (pix_x < im_w)
                & (pix_y >= 0)
                & (pix_y < im_h)
            )
            if not valid.any():
                continue

            depth_val = torch.zeros_like(cam_z)
            depth_val[valid] = depth[pix_y[valid], pix_x[valid]]
            depth_diff = depth_val - cam_z
            valid = valid & (depth_val > 0) & (depth_diff >= -self._trunc_margin)
            if not valid.any():
                continue

            dist = torch.clamp(depth_diff[valid] / self._trunc_margin, max=1.0)
            xv, yv, zv = x[valid], y[valid], z[valid]
            w_old = self._weight_vol[xv, yv, zv]
            tsdf_old = self._tsdf_vol[xv, yv, zv]
            w_new = w_old + float(obs_weight)
            self._weight_vol[xv, yv, zv] = w_new
            self._tsdf_vol[xv, yv, zv] = (tsdf_old * w_old + dist * float(obs_weight)) / w_new

    def get_volume(self):
        return self._tsdf_vol.detach().cpu().numpy(), self._weight_vol.detach().cpu().numpy()

    def get_surface_points(self, threshold=0.05, max_points=200000):
        tsdf, weights = self.get_volume()
        mask = (np.abs(tsdf) < threshold) & (weights > 0)
        coords = np.argwhere(mask)
        if coords.shape[0] > max_points:
            choice = np.random.choice(coords.shape[0], max_points, replace=False)
            coords = coords[choice]
        return coords.astype(np.float32) * self._voxel_size + self._vol_origin_np


class NGSTSDFPriorFuser:
    """GSPrior-style TSDF fusion for the vehicle model's surf-depth output."""

    def __init__(
        self,
        gaussians,
        render_func,
        pipe,
        sdf_trunc,
        bg_color=None,
        voxel_size=0.02,
        depth_trunc=5.0,
        alpha_threshold=0.05,
        bounds="gaussians",
        bounds_padding=0.10,
        bounds_quantile=0.05,
        chunk_size=262144,
        max_voxels=2_000_000,
    ):
        if bg_color is None:
            bg_color = [0, 0, 0]
        self.gaussians = gaussians
        self.render_func = render_func
        self.pipe = pipe
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.voxel_size = float(voxel_size)
        self.depth_trunc = float(depth_trunc)
        self.sdf_trunc = float(sdf_trunc)
        self.alpha_threshold = float(alpha_threshold)
        self.bounds = bounds
        self.bounds_padding = float(bounds_padding)
        self.bounds_quantile = float(bounds_quantile)
        self.chunk_size = int(chunk_size)
        self.max_voxels = int(max_voxels)

    @staticmethod
    def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
        Rt = np.zeros((4, 4), dtype=np.float32)
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        c2w = np.linalg.inv(Rt)
        cam_center = c2w[:3, 3]
        cam_center = (cam_center + translate) * scale
        c2w[:3, 3] = cam_center
        return np.linalg.inv(c2w).astype(np.float32)

    @staticmethod
    def get_extrinsic(viewpoint_cam):
        return NGSTSDFPriorFuser.getWorld2View2(
            viewpoint_cam.R,
            viewpoint_cam.T,
            translate=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
        )

    @staticmethod
    def get_intrinsic(viewpoint_cam):
        height = viewpoint_cam.image_height
        width = viewpoint_cam.image_width
        fx = width / (2.0 * math.tan(viewpoint_cam.FoVx / 2.0))
        fy = height / (2.0 * math.tan(viewpoint_cam.FoVy / 2.0))
        return np.array(
            [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _prepare_depth(self, render_out):
        if "surf_depth" not in render_out:
            raise KeyError("Vehicle render output must contain 'surf_depth' for TSDF fusion.")
        depth = render_out["surf_depth"].squeeze().detach().float().clone()
        depth[~torch.isfinite(depth)] = 0
        depth[depth <= 0] = 0
        depth[depth > self.depth_trunc] = 0
        if self.alpha_threshold > 0 and "rend_alpha" in render_out:
            alpha = render_out["rend_alpha"].squeeze().detach()
            depth[alpha < self.alpha_threshold] = 0
        return depth

    def _parse_explicit_bounds(self, bounds):
        if isinstance(bounds, str):
            parts = [p.strip() for p in bounds.split(",") if p.strip()]
            if len(parts) == 6:
                vals = np.asarray([float(p) for p in parts], dtype=np.float32)
                return vals.reshape(3, 2)
            return None
        arr = np.asarray(bounds, dtype=np.float32)
        return arr if arr.shape == (3, 2) else None

    def _gaussian_bounds(self):
        xyz = self.gaussians.get_xyz.detach()
        if xyz.numel() == 0:
            return np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32)
        q = min(max(self.bounds_quantile, 0.0), 0.45)
        lo = torch.quantile(xyz, q, dim=0).detach().cpu().numpy()
        hi = torch.quantile(xyz, 1.0 - q, dim=0).detach().cpu().numpy()
        extent = np.maximum(hi - lo, 1e-4)
        pad = np.maximum(self.bounds_padding, 0.08 * extent)
        return np.stack((lo - pad, hi + pad), axis=1).astype(np.float32)

    def _auto_frustum_bounds(self, viewpoint_stack, intrinsic):
        vol_bnds = np.zeros((3, 2), dtype=np.float32)
        for viewpoint_cam in tqdm(viewpoint_stack, desc="TSDF bounds"):
            out = self.render_func(viewpoint_cam, self.gaussians, self.pipe, self.background)
            depth = self._prepare_depth(out).detach().cpu().numpy()
            pose_c2w = np.linalg.inv(self.get_extrinsic(viewpoint_cam))
            view_frust_pts = get_view_frustum(depth, intrinsic, pose_c2w)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        if not np.isfinite(vol_bnds).all() or np.any(vol_bnds[:, 1] <= vol_bnds[:, 0]):
            return self._gaussian_bounds()
        return vol_bnds

    def _alpha_surface_bounds(self, viewpoint_stack, intrinsic, max_points_per_view=2048):
        points = []
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        for viewpoint_cam in tqdm(viewpoint_stack, desc="TSDF alpha bounds"):
            out = self.render_func(viewpoint_cam, self.gaussians, self.pipe, self.background)
            depth = self._prepare_depth(out)
            valid = depth > 0
            if not valid.any():
                continue
            ys, xs = torch.nonzero(valid, as_tuple=True)
            if ys.numel() > max_points_per_view:
                step = max(ys.numel() // max_points_per_view, 1)
                ys = ys[::step][:max_points_per_view]
                xs = xs[::step][:max_points_per_view]
            z = depth[ys, xs]
            x = (xs.float() - cx) * z / fx
            y = (ys.float() - cy) * z / fy
            cam_pts = torch.stack((x, y, z, torch.ones_like(z)), dim=1)
            pose_c2w = torch.from_numpy(np.linalg.inv(self.get_extrinsic(viewpoint_cam))).float().to(cam_pts.device)
            world_pts = cam_pts @ pose_c2w.T
            points.append(world_pts[:, :3].detach().cpu())
        if not points:
            return self._gaussian_bounds()
        pts = torch.cat(points, dim=0).float()
        q = min(max(self.bounds_quantile, 0.0), 0.45)
        lo = torch.quantile(pts, q, dim=0).numpy()
        hi = torch.quantile(pts, 1.0 - q, dim=0).numpy()
        extent = np.maximum(hi - lo, 1e-4)
        pad = np.maximum(self.bounds_padding, 0.08 * extent)
        return np.stack((lo - pad, hi + pad), axis=1).astype(np.float32)

    def _estimate_bounds(self, viewpoint_stack, intrinsic):
        explicit = self._parse_explicit_bounds(self.bounds)
        if explicit is not None:
            return explicit
        mode = str(self.bounds).lower()
        if mode == "auto":
            return self._auto_frustum_bounds(viewpoint_stack, intrinsic)
        if mode == "alpha_frustum":
            return self._alpha_surface_bounds(viewpoint_stack, intrinsic)
        if mode == "fixed":
            return np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32)
        return self._gaussian_bounds()

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        if len(viewpoint_stack) == 0:
            raise ValueError("TSDF fusion needs at least one camera.")
        intrinsic = self.get_intrinsic(viewpoint_stack[0])
        vol_bnds = self._estimate_bounds(viewpoint_stack, intrinsic)
        tsdf_volume = TorchTSDFVolume(
            vol_bnds,
            voxel_size=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            chunk_size=self.chunk_size,
            max_voxels=self.max_voxels,
        )
        print("TSDF bounds:\n", vol_bnds)

        for viewpoint_cam in tqdm(viewpoint_stack, desc="TSDF fusion"):
            out = self.render_func(viewpoint_cam, self.gaussians, self.pipe, self.background)
            depth = self._prepare_depth(out)
            pose_w2c = self.get_extrinsic(viewpoint_cam)
            pose_c2w = np.linalg.inv(pose_w2c)
            tsdf_volume.integrate(
                depth.detach().cpu().numpy(),
                intrinsic,
                pose_c2w,
                obs_weight=1.0,
            )

        tsdf, _ = tsdf_volume.get_volume()
        query_points = tsdf_volume.get_surface_points()
        return query_points, tsdf, tsdf_volume.vol_origin, tsdf_volume.voxel_size, None
