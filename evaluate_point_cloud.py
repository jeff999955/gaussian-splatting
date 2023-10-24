import glob
import os

import numpy as np
from icecream import ic
from plyfile import PlyData, PlyElement

from arguments import argparser
from scene.gaussian_model import GaussianModel


def load_gaussians(args, path):
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(path, cpu=True)
    return gaussians


if __name__ == "__main__":
    parser = argparser(desc="Training script parameters")

    args = parser.parse_args()
    ic(vars(args))

    for p in sorted(
        glob.glob(os.path.join(args.source_path, "iteration_*", "point_cloud.ply"))
    ):
        img_path = os.path.join(args.source_path, f"{p.split('/')[-2]}.png")
        print(p.split("/")[-2])
        gaussians = load_gaussians(args, p)
        ic(gaussians.get_opacity.shape)
        ic(np.sum(np.isnan(gaussians.get_opacity.detach().cpu().numpy())))

        ic(gaussians.get_opacity.shape[0] * gaussians.get_opacity.shape[1])

        # Plot the distribution of opacity values through line graph
        import matplotlib.pyplot as plt

        plt.hist(gaussians.get_opacity.detach().cpu().numpy().flatten(), bins=100)
        plt.savefig(img_path)
        plt.close()
