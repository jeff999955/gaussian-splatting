from plyfile import PlyData, PlyElement

from scene.gaussian_model import GaussianModel


def load_gaussians(path):
    gaussians = GaussianModel()

    gaussians.load_ply(path, cpu=True)

    return gaussians


if __name__ == "__main__":
    import sys

    load_gaussians(sys.argv[1])
