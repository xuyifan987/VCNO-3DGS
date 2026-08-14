#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, use_neural_appearance=False, hidden_neuron=6):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.use_neural_appearance = bool(use_neural_appearance)
        self.input_dim = 5
        self.output_dim = 3
        self.hidden_neuron = int(hidden_neuron)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._W1 = torch.empty(0)
        self._b1 = torch.empty(0)
        self._W2 = torch.empty(0)
        self._b2 = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._W1,
            self._b1,
            self._W2,
            self._b2,
            self.use_neural_appearance,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._W1,
        self._b1,
        self._W2,
        self._b2,
        self.use_neural_appearance,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_layer_1_weight(self):
        return self._W1, self._b1

    @property
    def get_layer_2_weight(self):
        return self._W2, self._b2
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        if self.use_neural_appearance:
            W1 = (2.0 * torch.rand((fused_color.shape[0], self.hidden_neuron, self.input_dim)).float().cuda() - 1) / self.input_dim
            b1 = (2.0 * torch.rand((fused_color.shape[0], self.hidden_neuron)).float().cuda() - 1) / self.input_dim
            W2 = math.sqrt(6.0 / self.input_dim) * (2.0 * torch.rand((fused_color.shape[0], self.output_dim, self.hidden_neuron)).float().cuda() - 1)
            b2 = math.sqrt(6.0 / self.input_dim) * (2.0 * torch.rand((fused_color.shape[0], self.output_dim)).float().cuda() - 1)
            self._W1 = nn.Parameter(W1.contiguous().requires_grad_(True))
            self._b1 = nn.Parameter(b1.contiguous().requires_grad_(True))
            self._W2 = nn.Parameter(W2.contiguous().requires_grad_(True))
            self._b2 = nn.Parameter(b2.contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.use_neural_appearance:
            l.extend([
                {'params': [self._W1], 'lr': training_args.feature_lr, "name": "w1"},
                {'params': [self._b1], 'lr': training_args.feature_lr, "name": "b1"},
                {'params': [self._W2], 'lr': training_args.feature_lr, "name": "w2"},
                {'params': [self._b2], 'lr': training_args.feature_lr, "name": "b2"},
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        if self.use_neural_appearance:
            for i in range(self._W1.shape[1] * self._W1.shape[2]):
                l.append('w1_{}'.format(i))
            for i in range(self._b1.shape[1]):
                l.append('b1_{}'.format(i))
            for i in range(self._W2.shape[1] * self._W2.shape[2]):
                l.append('w2_{}'.format(i))
            for i in range(self._b2.shape[1]):
                l.append('b2_{}'.format(i))
        else:
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        if self.use_neural_appearance:
            f_dc = None
            f_rest = None
            w1 = self._W1.detach().flatten(start_dim=1).contiguous().cpu().numpy()
            b1 = self._b1.detach().contiguous().cpu().numpy()
            w2 = self._W2.detach().flatten(start_dim=1).contiguous().cpu().numpy()
            b2 = self._b2.detach().contiguous().cpu().numpy()
        else:
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.use_neural_appearance:
            attributes = np.concatenate((xyz, normals, w1, b1, w2, b2, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        property_names = [p.name for p in plydata.elements[0].properties]
        neural_ply = any(name.startswith("w1_") for name in property_names)

        if neural_ply:
            self.use_neural_appearance = True
            w1 = np.zeros((xyz.shape[0], self.hidden_neuron, self.input_dim))
            for i in range(self.hidden_neuron):
                for j in range(self.input_dim):
                    w1[:, i, j] = np.asarray(plydata.elements[0][f"w1_{i * self.input_dim + j}"])
            b1 = np.zeros((xyz.shape[0], self.hidden_neuron))
            for i in range(self.hidden_neuron):
                b1[:, i] = np.asarray(plydata.elements[0][f"b1_{i}"])
            w2 = np.zeros((xyz.shape[0], self.output_dim, self.hidden_neuron))
            for i in range(self.output_dim):
                for j in range(self.hidden_neuron):
                    w2[:, i, j] = np.asarray(plydata.elements[0][f"w2_{i * self.hidden_neuron + j}"])
            b2 = np.zeros((xyz.shape[0], self.output_dim))
            for i in range(self.output_dim):
                b2[:, i] = np.asarray(plydata.elements[0][f"b2_{i}"])
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        else:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        if self.use_neural_appearance:
            self._W1 = nn.Parameter(torch.tensor(w1, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._W2 = nn.Parameter(torch.tensor(w2, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if self.use_neural_appearance:
            self._W1 = optimizable_tensors["w1"]
            self._b1 = optimizable_tensors["b1"]
            self._W2 = optimizable_tensors["w2"]
            self._b2 = optimizable_tensors["b2"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                              new_W1=None, new_b1=None, new_W2=None, new_b2=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.use_neural_appearance:
            d.update({
                "w1": new_W1,
                "b1": new_b1,
                "w2": new_W2,
                "b2": new_b2,
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if self.use_neural_appearance:
            self._W1 = optimizable_tensors["w1"]
            self._b1 = optimizable_tensors["b1"]
            self._W2 = optimizable_tensors["w2"]
            self._b2 = optimizable_tensors["b2"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_W1 = self._W1[selected_pts_mask].repeat(N, 1, 1) if self.use_neural_appearance else None
        new_b1 = self._b1[selected_pts_mask].repeat(N, 1) if self.use_neural_appearance else None
        new_W2 = self._W2[selected_pts_mask].repeat(N, 1, 1) if self.use_neural_appearance else None
        new_b2 = self._b2[selected_pts_mask].repeat(N, 1) if self.use_neural_appearance else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_W1, new_b1, new_W2, new_b2)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_W1 = self._W1[selected_pts_mask] if self.use_neural_appearance else None
        new_b1 = self._b1[selected_pts_mask] if self.use_neural_appearance else None
        new_W2 = self._W2[selected_pts_mask] if self.use_neural_appearance else None
        new_b2 = self._b2[selected_pts_mask] if self.use_neural_appearance else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                   new_W1, new_b1, new_W2, new_b2)

    def _detail_routed_densify_score(
        self,
        errors,
        reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
    ):
        score = errors.squeeze().clamp_min(0.0)
        if reliability is None or detail_weight <= 0.0:
            return score, None
        reliability = reliability.squeeze().clamp(0.0, 1.0)
        if min_reliability > 0.0:
            reliability = torch.where(reliability >= min_reliability, reliability, torch.zeros_like(reliability))
        detail_gate = reliability.pow(max(float(reliability_power), 0.0))
        return score * (1.0 + float(detail_weight) * detail_gate), detail_gate

    def _selected_detail_strength(self, score, selected_mask):
        if selected_mask.sum() == 0:
            return torch.zeros((0, 1), device=self._xyz.device)
        selected_score = score[selected_mask].clamp_min(0.0)
        strength = selected_score / selected_score.max().clamp_min(1e-6)
        return strength.clamp(0.0, 1.0).unsqueeze(-1)

    def densify_and_split_for_error_based(
        self,
        errors,
        error_threshold,
        max_primitive_num,
        scene_extent,
        N=2,
        reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
        scale_shrink=0.0,
    ):
        n_init_points = self.get_xyz.shape[0]
        if max_primitive_num is None or max_primitive_num <= 0:
            max_primitive_num = n_init_points + max(1, int(n_init_points * 0.05))
        max_new_points = min(int(n_init_points * 0.05), max_primitive_num - n_init_points)
        if max_new_points <= 0:
            return

        padded_error = torch.zeros((n_init_points), device="cuda")
        padded_error[:errors.shape[0]] = errors.squeeze()
        padded_reliability = None
        if reliability is not None:
            padded_reliability = torch.ones((n_init_points), device="cuda")
            padded_reliability[:reliability.shape[0]] = reliability.squeeze()
        score, _ = self._detail_routed_densify_score(
            padded_error,
            padded_reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
        )
        selected_pts_mask = torch.logical_and(
            score >= error_threshold,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )
        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze()
        if selected_indices.numel() == 0:
            return
        if selected_indices.dim() == 0:
            selected_indices = selected_indices[None]
        selected_errors = score[selected_indices]
        sorted_indices = selected_indices[torch.argsort(selected_errors, descending=True)]
        sorted_indices = sorted_indices[:max_new_points]
        final_mask = torch.zeros_like(selected_pts_mask)
        final_mask[sorted_indices] = True

        detail_strength = self._selected_detail_strength(score, final_mask)
        stds = self.get_scaling[final_mask].repeat(N, 1)
        stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(self._rotation[final_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[final_mask].repeat(N, 1)
        shrink = (1.0 + max(float(scale_shrink), 0.0) * detail_strength).repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[final_mask].repeat(N, 1) / (0.8 * N * shrink))
        new_rotation = self._rotation[final_mask].repeat(N, 1)
        new_features_dc = self._features_dc[final_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[final_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[final_mask].repeat(N, 1)
        new_W1 = self._W1[final_mask].repeat(N, 1, 1) if self.use_neural_appearance else None
        new_b1 = self._b1[final_mask].repeat(N, 1) if self.use_neural_appearance else None
        new_W2 = self._W2[final_mask].repeat(N, 1, 1) if self.use_neural_appearance else None
        new_b2 = self._b2[final_mask].repeat(N, 1) if self.use_neural_appearance else None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
            new_W1, new_b1, new_W2, new_b2,
        )
        prune_filter = torch.cat((final_mask, torch.zeros(N * final_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_for_error_based(
        self,
        errors,
        error_threshold,
        max_primitive_num,
        scene_extent,
        reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
        opacity_boost=0.0,
    ):
        n_init_points = self.get_xyz.shape[0]
        if max_primitive_num is None or max_primitive_num <= 0:
            max_primitive_num = n_init_points + max(1, int(n_init_points * 0.05))
        max_new_points = min(int(n_init_points * 0.05), max_primitive_num - n_init_points)
        if max_new_points <= 0:
            return
        score, _ = self._detail_routed_densify_score(
            errors,
            reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
        )
        selected_pts_mask = torch.logical_and(
            score >= error_threshold,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze()
        if selected_indices.numel() == 0:
            return
        if selected_indices.dim() == 0:
            selected_indices = selected_indices[None]
        selected_errors = score[selected_indices]
        sorted_indices = selected_indices[torch.argsort(selected_errors, descending=True)]
        sorted_indices = sorted_indices[:max_new_points]
        final_mask = torch.zeros_like(selected_pts_mask)
        final_mask[sorted_indices] = True
        detail_strength = self._selected_detail_strength(score, final_mask)

        new_xyz = self._xyz[final_mask]
        new_features_dc = self._features_dc[final_mask]
        new_features_rest = self._features_rest[final_mask]
        new_opacities = self._opacity[final_mask]
        if detail_strength.numel() > 0 and opacity_boost > 0:
            alpha = self.opacity_activation(new_opacities)
            alpha = alpha + float(opacity_boost) * detail_strength.to(alpha.device) * (1.0 - alpha)
            new_opacities = self.inverse_opacity_activation(alpha.clamp(1e-6, 1.0 - 1e-6))
        new_scaling = self._scaling[final_mask]
        new_rotation = self._rotation[final_mask]
        new_W1 = self._W1[final_mask] if self.use_neural_appearance else None
        new_b1 = self._b1[final_mask] if self.use_neural_appearance else None
        new_W2 = self._W2[final_mask] if self.use_neural_appearance else None
        new_b2 = self._b2[final_mask] if self.use_neural_appearance else None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
            new_W1, new_b1, new_W2, new_b2,
        )

    def error_based_densify_and_prune(
        self,
        gaussian_error,
        max_primitive_num,
        error_threshhold,
        min_opacity,
        extent,
        max_screen_size,
        gaussian_reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
        scale_shrink=0.0,
        opacity_boost=0.0,
    ):
        self.densify_and_clone_for_error_based(
            gaussian_error,
            error_threshhold,
            max_primitive_num,
            extent,
            reliability=gaussian_reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
            opacity_boost=opacity_boost,
        )
        self.densify_and_split_for_error_based(
            gaussian_error,
            error_threshhold,
            max_primitive_num,
            extent,
            reliability=gaussian_reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
            scale_shrink=scale_shrink,
        )
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        cap_stats = self.enforce_max_primitives(max_primitive_num)
        torch.cuda.empty_cache()
        return cap_stats

    def enforce_max_primitives(self, max_primitive_num):
        if max_primitive_num is None or max_primitive_num <= 0:
            return {"hard_cap_pruned": 0, "points": int(self._xyz.shape[0])}
        n_points = int(self._xyz.shape[0])
        if n_points <= max_primitive_num:
            return {"hard_cap_pruned": 0, "points": n_points}
        with torch.no_grad():
            score = self.get_opacity.detach().squeeze()
            score = score + 1e-6 * torch.rand_like(score)
            keep_idx = torch.topk(score, k=int(max_primitive_num), largest=True, sorted=False).indices
            keep_mask = torch.zeros((n_points,), device=self._xyz.device, dtype=torch.bool)
            keep_mask[keep_idx] = True
            self.prune_points(~keep_mask)
        return {"hard_cap_pruned": n_points - int(max_primitive_num), "points": int(self._xyz.shape[0])}

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_primitive_num=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.enforce_max_primitives(max_primitive_num)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
