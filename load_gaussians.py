from plyfile import PlyData, PlyElement

from scene.gaussian_model import GaussianModel
from arguments import argparser
from icecream import ic

import numpy as np

def load_gaussians(args,path):
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(path, cpu=True)
    return gaussians


if __name__ == "__main__":
    parser = argparser(desc="Training script parameters")
    args = parser.parse_args()

    ic(vars(args))
    gaussians = load_gaussians(args,args.source_path)
    ic(gaussians.get_xyz.shape)
    ic(np.sum(np.isnan(gaussians.get_xyz.detach().cpu().numpy())))
    
    ic(gaussians.get_xyz.shape[0] * gaussians.get_xyz.shape[1])
