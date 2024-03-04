import argparse

from plyfile import PlyData
import numpy as np


def get_bbox(path):
    if not path.endswith(".ply"):
        raise ValueError("Path must be a .ply file")
    

    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    x, y, z = vertices["x"], vertices["y"], vertices["z"]

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    min_z, max_z = min(z), max(z)

    c_x, c_y, c_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
    m_x, m_y, m_z = np.mean(x), np.mean(y), np.mean(z)

    # Print with 2 decimal places
    print(f"bbox: ({min_x:.2f}, {min_y:.2f}, {min_z:.2f}) - ({max_x:.2f}, {max_y:.2f}, {max_z:.2f})")
    print(f"center: ({c_x:.2f}, {c_y:.2f}, {c_z:.2f})")
    print(f"centroid: ({m_x:.2f}, {m_y:.2f}, {m_z:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get number of points in a point cloud')
    parser.add_argument("file", type=str, help="point cloud file path")

    args = parser.parse_args()
    get_bbox(args.file)