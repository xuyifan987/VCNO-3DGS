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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.down_sample = True # Sampling initial point
        self.init_point_num = 5000 # The number of initial point
        self.densification_camera_num = 20 # The number of camera used for error based densification
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 20_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05
        # Paper-level module switches. Low-level car_fusion options remain
        # available for controlled ablations and parameter sweeps. The
        # vehicle_* names are the publication-facing module interface; the
        # older ours_* names are kept as backward-compatible aliases.
        self.vehicle_appearance = False
        self.vehicle_allocation = False
        self.vehicle_geometry = False
        self.vehicle_full = False
        self.ours_appearance = False
        self.ours_frequency = False
        self.ours_geometry = False
        self.ours_full = False
        self.car_fusion = False
        self.car_fusion_reliability_power = 1.5
        self.car_fusion_residual_temperature = 0.20
        self.car_fusion_reliability_kernel = 9
        self.car_fusion_reliability_detail_preserve = 0.25
        self.car_fusion_reliability_edge_boost = 0.50
        self.car_fusion_error_threshold = 0.012
        self.car_fusion_uncertain_prune = False
        self.car_fusion_uncertain_prune_threshold = 0.02
        self.car_fusion_hard_view_sampling = False
        self.car_fusion_hard_view_start_iter = 1000
        self.car_fusion_hard_view_warmup = 2000
        self.car_fusion_hard_view_momentum = 0.90
        self.car_fusion_hard_view_power = 2.0
        self.car_fusion_hard_view_max_prob_mult = 4.0
        self.car_fusion_hard_view_uniform_mix = 0.35
        self.car_fusion_hard_view_ssim_weight = 0.50
        self.car_fusion_hard_view_detail_boost = 0.35
        self.car_fusion_hard_view_detail_power = 1.0
        self.car_fusion_hard_view_detail_min_reliability = 0.25
        self.car_fusion_metric_adaptive_loss = False
        self.car_fusion_metric_adaptive_start_iter = 1000
        self.car_fusion_metric_adaptive_warmup = 2000
        self.car_fusion_metric_adaptive_max_boost = 0.35
        self.car_fusion_metric_adaptive_target_ssim_share = 0.50
        self.car_fusion_mse_loss = False
        self.car_fusion_mse_weight = 0.10
        self.car_fusion_mse_start_iter = 7000
        self.car_fusion_mse_warmup = 3000
        self.car_fusion_flat_mse_loss = False
        self.car_fusion_flat_mse_weight = 0.75
        self.car_fusion_flat_mse_start_iter = 5000
        self.car_fusion_flat_mse_warmup = 3000
        self.car_fusion_flat_mse_edge_suppression = 1.5
        self.car_fusion_flat_mse_min_weight = 0.15
        self.car_fusion_flat_mse_residual_gain = 1.0
        self.car_fusion_flat_mse_residual_kernel = 7
        self.car_fusion_flat_mse_reliability_power = 0.0
        self.car_fusion_detail_routed_densify = False
        self.car_fusion_visibility_aware_densify = False
        self.car_fusion_densify_reliability_error = True
        self.car_fusion_densify_raw_error = False
        self.car_fusion_densify_reliability_floor = 0.0
        self.car_fusion_densify_reliability_floor_start = -1.0
        self.car_fusion_densify_reliability_floor_end = -1.0
        self.car_fusion_densify_reliability_floor_start_iter = 500
        self.car_fusion_densify_reliability_floor_warmup = 2000
        self.car_fusion_densify_adaptive_floor_strength = 0.0
        self.car_fusion_densify_adaptive_floor_power = 1.0
        self.car_fusion_densify_adaptive_floor_detail_power = 0.0
        self.car_fusion_densify_detail_weight = 1.5
        self.car_fusion_densify_reliability_power = 1.0
        self.car_fusion_densify_min_reliability = 0.05
        self.car_fusion_densify_scale_shrink = 0.35
        self.car_fusion_densify_opacity_boost = 0.10
        self.car_fusion_densify_min_views = 2
        self.car_fusion_densify_visibility_min_effect = 1e-5
        self.car_fusion_densify_under_supported_scale = 0.25
        self.car_fusion_densify_consistency_floor = 0.25
        self.car_fusion_densify_confidence_blend = 0.0
        self.car_fusion_densify_view_coverage = False
        self.car_fusion_densify_coverage_floor = 0.35
        self.car_fusion_densify_coverage_power = 1.0
        self.car_fusion_structure_loss = False
        self.car_fusion_structure_weight = 0.01
        self.car_fusion_structure_start_iter = 1000
        self.car_fusion_structure_warmup = 1500
        self.car_fusion_structure_scales = "2,4"
        self.car_fusion_structure_reliability_power = 1.5
        self.car_fusion_structure_min_reliability = 0.10
        self.car_fusion_structure_edge_boost = 0.50
        self.car_fusion_detail_photometric_loss = False
        self.car_fusion_detail_photometric_weight = 0.015
        self.car_fusion_detail_photometric_start_iter = 1000
        self.car_fusion_detail_photometric_warmup = 2000
        self.car_fusion_detail_photometric_reliability_power = 1.5
        self.car_fusion_detail_photometric_min_reliability = 0.15
        self.car_fusion_detail_photometric_edge_boost = 0.50
        self.car_fusion_detail_photometric_residual_stop = 0.20
        self.car_fusion_detail_photometric_residual_floor = 0.0
        self.car_fusion_detail_photometric_loss_type = "charbonnier"
        self.car_fusion_detail_ssim_loss = False
        self.car_fusion_detail_ssim_weight = 0.02
        self.car_fusion_detail_ssim_start_iter = 1000
        self.car_fusion_detail_ssim_warmup = 2000
        self.car_fusion_detail_ssim_window = 11
        self.car_fusion_detail_ssim_reliability_power = 1.2
        self.car_fusion_detail_ssim_min_reliability = 0.10
        self.car_fusion_detail_ssim_edge_boost = 0.75
        self.car_fusion_contrast_loss = False
        self.car_fusion_contrast_weight = 0.01
        self.car_fusion_contrast_start_iter = 1000
        self.car_fusion_contrast_warmup = 2000
        self.car_fusion_contrast_window = 9
        self.car_fusion_contrast_scales = "1,2,4"
        self.car_fusion_contrast_reliability_power = 1.2
        self.car_fusion_contrast_min_reliability = 0.10
        self.car_fusion_contrast_edge_boost = 0.75
        self.car_fusion_contrast_mean_weight = 0.25
        self.car_fusion_surface_prior = False
        self.car_fusion_surface_prior_weight = 0.02
        self.car_fusion_surface_prior_start = 2000
        self.car_fusion_surface_prior_warmup = 1500
        self.car_fusion_surface_prior_downsample = 4
        self.car_fusion_surface_prior_momentum = 0.95
        self.car_fusion_surface_prior_min_reliability = 0.10
        self.car_fusion_tsdf_prior = False
        self.car_fusion_tsdf_iters = "5000,10000"
        self.car_fusion_tsdf_truncs = "0.08,0.04"
        self.car_fusion_tsdf_voxel_size = 0.02
        self.car_fusion_tsdf_depth_trunc = 5.0
        self.car_fusion_tsdf_alpha_threshold = 0.05
        self.car_fusion_tsdf_bounds = "gaussians"
        self.car_fusion_tsdf_bounds_padding = 0.10
        self.car_fusion_tsdf_bounds_quantile = 0.05
        self.car_fusion_tsdf_max_voxels = 2000000
        self.car_fusion_tsdf_chunk_size = 262144
        self.car_fusion_tsdf_pull_interval = 100
        self.car_fusion_tsdf_pull_weight = 0.0001
        self.car_fusion_tsdf_max_pull = 0.002
        self.car_fusion_tsdf_pull_confidence_gate = False
        self.car_fusion_tsdf_pull_min_reliability = 0.10
        self.car_fusion_tsdf_pull_min_tsdf_confidence = 0.02
        self.car_fusion_tsdf_pull_gate_power = 1.0
        self.car_fusion_tsdf_pull_detail_suppression = 0.0
        self.car_fusion_tsdf_prune = False
        self.car_fusion_tsdf_outside = 0.999
        self.car_fusion_tsdf_prune_opacity = 0.10
        self.car_fusion_adaptive_gabor = False
        self.car_fusion_adaptive_gabor_strength = 0.75
        self.car_fusion_adaptive_gabor_max_log_scale = 0.70
        self.car_fusion_adaptive_gabor_view_strength = 0.50
        self.car_fusion_adaptive_gabor_phase_strength = 0.25
        self.car_fusion_adaptive_gabor_residual_strength = 0.15
        self.car_fusion_adaptive_gabor_residual_detail_power = 0.0
        self.car_fusion_adaptive_gabor_residual_scale_power = 0.0
        self.car_fusion_adaptive_gabor_residual_scale_quantile = 0.5
        self.car_fusion_adaptive_gabor_residual_scale_min_gate = 0.0
        self.car_fusion_adaptive_gabor_detail_weight = 0.75
        self.car_fusion_adaptive_gabor_detail_interval = 4
        self.car_fusion_adaptive_gabor_detail_momentum = 0.90
        self.car_fusion_adaptive_gabor_detail_reliability_floor = 0.10
        self.car_fusion_adaptive_gabor_detail_adaptive_floor_strength = 0.20
        self.car_fusion_adaptive_gabor_detail_adaptive_floor_power = 1.0
        self.car_fusion_adaptive_gabor_detail_adaptive_floor_detail_power = 1.0
        self.car_fusion_adaptive_gabor_start_iter = 1500
        self.car_fusion_adaptive_gabor_warmup = 1000
        self.car_fusion_adaptive_gabor_preserve_base = False
        self.car_fusion_neural_output_mode = "sigmoid"
        self.car_fusion_neural_output_delta_scale = 1.5
        self.car_fusion_appearance_backend = "torch_mlp"
        self.car_fusion_adaptive_gabor_lr_scale = 0.50
        self.car_fusion_adaptive_gabor_reg = 0.0001
        self.car_fusion_adaptive_gabor_min_confidence = 0.15
        self.car_fusion_adaptive_gabor_reliability_momentum = 0.90
        self.car_fusion_adaptive_gabor_tsdf_weight = 0.50
        self.car_fusion_adaptive_gabor_tsdf_default_confidence = 0.0

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.max_primitive_num = 20_000 # Primitive number limitation
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")


def apply_ours_module_presets(args):
    """Expand paper-level modules into existing implementation flags.

    Module A: vehicle_appearance
        Neural per-primitive appearance representation.
    Module B: vehicle_allocation
        Reliability and multi-view visibility guided primitive allocation.
    Module C: vehicle_geometry
        Confidence-gated surface consistency and TSDF geometry control.
    """
    if args.ours_appearance:
        args.vehicle_appearance = True
    if args.ours_frequency:
        args.vehicle_allocation = True
    if args.ours_geometry:
        args.vehicle_geometry = True
    if args.ours_full:
        args.vehicle_full = True

    if args.vehicle_full:
        args.vehicle_appearance = True
        args.vehicle_allocation = True
        args.vehicle_geometry = True

    enabled = []
    if args.vehicle_appearance:
        enabled.append("vehicle_appearance")
        args.car_fusion = True
        # Publication-facing appearance module: reliable detail supervision
        # activates the independent residual-adapted vehicle appearance MLP.
        args.car_fusion_adaptive_gabor = True
        args.car_fusion_detail_photometric_loss = True
        args.car_fusion_detail_ssim_loss = True
        args.car_fusion_structure_loss = True
        args.car_fusion_reliability_detail_preserve = max(args.car_fusion_reliability_detail_preserve, 0.35)
        args.car_fusion_reliability_edge_boost = max(args.car_fusion_reliability_edge_boost, 0.75)

    if args.vehicle_allocation:
        enabled.append("vehicle_allocation")
        args.car_fusion = True
        args.car_fusion_detail_routed_densify = True
        args.car_fusion_visibility_aware_densify = True
        args.car_fusion_densify_confidence_blend = max(args.car_fusion_densify_confidence_blend, 0.20)
        if args.car_fusion_densify_reliability_floor_start < 0:
            args.car_fusion_densify_reliability_floor_start = 0.20
        if args.car_fusion_densify_reliability_floor_end < 0:
            args.car_fusion_densify_reliability_floor_end = 0.05
        args.car_fusion_densify_reliability_floor_start_iter = min(args.car_fusion_densify_reliability_floor_start_iter, 500)
        args.car_fusion_densify_reliability_floor_warmup = max(args.car_fusion_densify_reliability_floor_warmup, 2500)
        args.car_fusion_densify_adaptive_floor_strength = max(args.car_fusion_densify_adaptive_floor_strength, 0.25)
        args.car_fusion_densify_adaptive_floor_detail_power = max(args.car_fusion_densify_adaptive_floor_detail_power, 1.0)

    if args.vehicle_geometry:
        enabled.append("vehicle_geometry")
        args.car_fusion = True
        args.car_fusion_tsdf_prior = True
        args.car_fusion_tsdf_iters = "500,1500,2500,4000,7000,10000,15000"
        args.car_fusion_tsdf_truncs = "0.08,0.06,0.05,0.04,0.035,0.03,0.025"
        args.car_fusion_tsdf_bounds = "alpha_frustum"
        args.car_fusion_tsdf_bounds_quantile = 0.05
        args.car_fusion_tsdf_pull_confidence_gate = True
        args.car_fusion_tsdf_pull_weight = max(args.car_fusion_tsdf_pull_weight, 0.00015)
        args.car_fusion_tsdf_pull_detail_suppression = max(args.car_fusion_tsdf_pull_detail_suppression, 0.25)

    return enabled

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
