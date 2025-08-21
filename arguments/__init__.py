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
import json
from argparse import ArgumentParser, Namespace
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
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
        self.sh_degree = 3
        self.root_out = "./output"
        self.exp_name = ""
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.points_pcl_suffix: str = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
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
        self.lambda_l1 = 0.8
        self.lambda_l2 = 0.0
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.increase_SH_deg_every = 1000
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


class GSGNParams(ParamGroup):
    def __init__(self, parser):
        self.use_double_precision: bool = False
        self.trust_region_radius: float = 1.0
        self.min_trust_region_radius: float = 1e-4
        self.max_trust_region_radius: float = 1e4
        self.radius_decrease_factor: float = 2.0
        self.max_grad_norm: float = 1e1
        self.min_lm_diagonal: float = 1e0
        self.max_lm_diagonal: float = 1e6
        self.min_relative_decrease: float = 1e-5
        self.function_tolerance: float = 0.000001
        self.pcg_max_iter: int = 100
        self.pcg_rtol: float = 1e-6
        self.pcg_atol: float = 0.0
        self.pcg_gradient_descent_every: int = -1
        self.pcg_explicit_residual_every: int = -1
        self.pcg_verbose: bool = False
        self.perc_images_in_line_search: float = 1.0
        self.line_search_initial_gamma: float = 1.0
        self.line_search_alpha: float = 0.7
        self.line_search_use_maximum_gamma: bool = False
        self.line_search_maximum_gamma: float = -1.0
        self.line_search_gamma_reset_interval: int = 100
        self.num_sgd_iterations_before_gn: int = 0
        self.num_sgd_iterations_between_gn: int = 0
        self.sgd_between_gn_every: int = 0
        self.num_sgd_iterations_after_gn: int = 0
        self.image_subsample_size: int = -1
        self.image_subsample_n_iters: int = 1
        self.image_subsample_frame_selection_mode: Literal["random", "strided"] = "strided"
        self.average_pcg_mode: Literal["mean", "max", "diag_jtj"] = "diag_jtj"
        self.scale_fac_xyz: float = 1.0
        self.scale_fac_opacity: float = 1.0
        self.scale_fac_rotation: float = 1.0
        self.scale_fac_scale: float = 1.0
        self.scale_fac_features_dc: float = 1.0
        self.scale_fac_features_rest: float = 1.0
        self.compute_huber_weights: bool = True
        self.huber_c: float = 0.1
        self.compute_ssim_weights: bool = True
        self.ssim_residual_scale_factor: float = 0.25
        super().__init__(parser, "GN Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")

    if not os.path.exists(cfgfilepath):
        cfgfilepath = cfgfilepath + ".json"
        merged_dict = {}
        with open(cfgfilepath) as cfg_file:
            cfg = json.load(cfg_file)
            for d in cfg.values():
                for k, v in d.items():
                    merged_dict[k] = v
    else:
        try:
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
