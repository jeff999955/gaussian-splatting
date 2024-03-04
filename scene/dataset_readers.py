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

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from icecream import ic
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.loadCalibration import loadCalibrationCameraToPose
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    """
    Represents the camera information for a single image in the dataset.

    Attributes:
        uid (int): The unique identifier for the camera.
        R (np.array): The rotation matrix of the camera.
        T (np.array): The translation vector of the camera.
        FovY (np.array): The vertical field of view of the camera.
        FovX (np.array): The horizontal field of view of the camera.
        image (Any): Do no use this field directly, it is lazily loaded in the training loop.
        image_path (str): The path to the image file.
        image_name (str): The name of the image file.
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.
    """

    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    args, cam_extrinsics, cam_intrinsics, images_folder
) -> List[CameraInfo]:
    cam_infos = []
    pose_path = None
    if args.use_ground_truth_pose:
        print("Using ground truth pose")
        pose_path = args.pose_path
        # trans, scale = align_trajectory(pose_path, cam_extrinsics)

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id

        if args.use_ground_truth_pose:
            # TODO: Find how to convert C2W to W2C from ScanNet
            c2w = np.loadtxt(os.path.join(pose_path, extr.name.replace(".jpg", ".txt")))

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
        else:
            # Colmap outputs the world-to-camera transform
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=None,  # Lazily load the image in training loop
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path, mask=False):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    if mask:
        lb, ub = [-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]
        positions = positions[
            np.prod(np.logical_and((positions > lb), (positions < ub)), axis=-1)
        ]
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except:
        normals = np.zeros_like(positions)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(args, path, images, eval, llffhold=8) -> SceneInfo:
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted: List[CameraInfo] = readColmapCameras(
        args=args,
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos: List[CameraInfo] = []
    test_cam_infos: List[CameraInfo] = []

    if eval:
        if args.split_setting == "pointnerf":
            step = 5
            train_cam_infos = cam_infos[::step]
            test_cam_infos = [*cam_infos]
        elif args.split_setting == "mipnerf":
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold != 0
            ]
            test_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold == 0
            ]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print("Train images: ", len(train_cam_infos))
    print("Test  images: ", len(test_cam_infos))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    if args.use_ground_truth_pose:
        ply_path = glob.glob(os.path.join(path, "pcd.ply"))
        if not len(ply_path):
            raise FileNotFoundError
        ply_path = ply_path[0]

        print(f"Using ground truth point cloud from {ply_path}")
    # TODO: Remove this, this is just for dense map debugging
    ply_path = os.path.join(path, "dense/0/fused.ply")
    try:
        pcd = fetchPly(ply_path, mask=args.use_ground_truth_pose)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readScanNetCameras(path, image_path):
    cam_infos = []

    intrinsics = np.loadtxt(os.path.join(path, "intrinsic", "intrinsic_color.txt"))

    sort_key = lambda x: int(os.path.basename(x).split(".")[0])

    poses = sorted(glob.glob(os.path.join(path, "pose", "*.txt")), key=sort_key)
    imgs = sorted(glob.glob(os.path.join(image_path, "*.jpg")), key=sort_key)

    # TODO: We should remove this divided by 2 hack
    x_scale = 620 / 1296
    y_scale = 440 / 968
    fx = intrinsics[0, 0] * x_scale
    fy = intrinsics[1, 1] * y_scale

    assert len(poses) == len(imgs)

    with Image.open(imgs[0]) as image:
        width, height = image.size

    for _, (pose, img) in enumerate(zip(poses, imgs)):
        c2w = np.loadtxt(pose)
        w2c = np.linalg.inv(c2w)

        if np.any(np.isnan(w2c)):
            ic(f"Something's wrong with {pose}")
            ic(c2w)
            ic(w2c)

            continue
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)

        image_name = os.path.basename(img).split(".")[0]
        cam_info = CameraInfo(
            uid=int(image_name),
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=None,  # Lazily load the image in training loop
            image_path=img,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    return cam_infos


def readScanNetInfo(args, path, eval, llffhold=8) -> SceneInfo:
    cam_infos = readScanNetCameras(path, args.images)

    ic(cam_infos[0])

    train_cam_infos: List[CameraInfo] = []
    test_cam_infos: List[CameraInfo] = []

    if eval:
        if args.split_setting == "pointnerf":
            step = 5
            train_cam_infos = cam_infos[::step]
            test_cam_infos = [*cam_infos]
        elif args.split_setting == "mipnerf":
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold != 0
            ]
            test_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold == 0
            ]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print("Train images: ", len(train_cam_infos))
    print("Test  images: ", len(test_cam_infos))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    plys = glob.glob(os.path.join(path, "*vh_clean.ply"))

    ic(plys)
    try:
        ply_path = plys[0]
        pcd = fetchPly(ply_path, mask=args.use_ground_truth_pose)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readKittiCameras(root_path, frame_list_path: str, step=10, kitti_cameras = ["00", "01"]):
    """
    frame_list_path: absolute path to the text file containing the list of path to images
    """

    assert os.path.isfile(frame_list_path)

    # TODO: read all images from frame_list_path
    with open(frame_list_path) as f:
        image_paths = f.readlines()

    image_paths = [
        os.path.join(frame_list_path.replace(".txt", ""), image_path).strip()
        for image_path in image_paths
    ]

    if len(kitti_cameras) == 2: # ["00", "01"]
        all_image_paths = image_paths + [path.replace("image_00", "image_01") for path in image_paths]
    else: # ["00"]
        all_image_paths = image_paths
    

    images_id = list(range(len(all_image_paths)))
    test_id_list = images_id[::step]
    train_id_list = list(set(images_id) - set(test_id_list))
    print("all_id_list", len(images_id))
    print(test_id_list)

    scan_name = all_image_paths[0].split("/")[-4]
    train_cam_infos, test_cam_infos = [], []

    # * Load files shared between cameras
    intrinsic_file = os.path.join(root_path, "calibration", "perspective.txt")
    with open(intrinsic_file) as f:
        intrinsics = f.readlines()

    # * Load the camera extrinsics into a list
    pose_file = os.path.join(root_path, "data_poses", scan_name, "poses.txt")
    poses = np.loadtxt(pose_file)
    frames, poses = poses[:, 0].astype(np.int32), np.reshape(poses[:, 1:], (-1, 3, 4))

    cam_to_pose = loadCalibrationCameraToPose(
        os.path.join(root_path, "calibration", "calib_cam_to_pose.txt")
    )

    idx = 0    
    for cam_id in kitti_cameras:
        # * Change if not working
        image_paths = list(
            map(lambda x: re.sub(r"image_(0[0-9])", f"image_{cam_id}", x), image_paths)
        )

        # * Load the camera intrinsics
        intrinsic_loaded = False
        for line in intrinsics:
            line = line.split(" ")
            if line[0] == f"P_rect_{cam_id}:":
                K = np.array(line[1:], dtype=np.float32)
                K = np.reshape(K, (3, 4))
                intrinsic_loaded = True
            elif line[0] == f"R_rect_{cam_id}:":
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == f"S_rect_{cam_id}:":
                width = int(float(line[1]))
                height = int(float(line[2]))

        assert intrinsic_loaded == True
        assert width > 0 and height > 0

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # * Load the camera extrinsics along with images
        assert f"image_{cam_id}" in cam_to_pose

        c2w_list = [0 for _ in range(frames.max() + 1)]
        for frame, pose in zip(frames, poses):
            pose = np.concatenate([pose, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
            c2w_list[frame] = (
                pose @ cam_to_pose.get(f"image_{cam_id}") @ np.linalg.inv(R_rect)
            )

        # * Load all images
        
        for _, img in enumerate(image_paths):
            frame_idx = int(img.split("/")[-1].split(".")[0])
            c2w = c2w_list[frame_idx]
            w2c = np.linalg.inv(c2w)

            if np.any(np.isnan(w2c)):
                ic(f"Something's wrong with {frame_idx}")
                ic(c2w)
                ic(w2c)

                continue
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # TODO: try different height and width
            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)

            cam_info = CameraInfo(
                uid=frame_idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=None,  # Lazily load the image in training loop
                image_path=img,
                image_name=str(frame_idx),
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy
            )

            if idx in train_id_list:
                train_cam_infos.append(cam_info)
            elif idx in test_id_list:
                test_cam_infos.append(cam_info)
            else:
                print("Something's wrong with the train/test split")
                print(f"Frame {frame_idx} not in train nor test list")
            idx += 1
    return train_cam_infos, test_cam_infos


def readKittiInfo(args, path, images_list, is_test=False) -> SceneInfo:
    train_cam_infos, test_cam_infos = readKittiCameras(path, images_list)
    if is_test:
        train_cam_infos = train_cam_infos + test_cam_infos
        test_cam_infos = readKittiCameras(path, images_list.replace("train_", "test_"), kitti_cameras=["00"])
        test_cam_infos = test_cam_infos[0] + test_cam_infos[1]
        test_cam_infos = sorted(test_cam_infos, key=lambda x: x.uid)
        
    print("Train images: ", len(train_cam_infos))
    print("Test  images: ", len(test_cam_infos))

    for cam in test_cam_infos:
        print(cam.image_path)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(
        path, "pcd", os.path.basename(images_list).replace("txt", "ply")
    )

    if args.random_init_pcd:
        # Read the original
        plydata = PlyData.read(ply_path)
        vertices = plydata["vertex"]
        x, y, z = vertices["x"], vertices["y"], vertices["z"]

        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        min_z, max_z = min(z), max(z)

        # Generate a random point cloud
        ply_path = os.path.join(path, "points3d.ply")
        num_pts = 400_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3))
        xyz[:, 0] = xyz[:, 0] * (max_x - min_x) + min_x
        xyz[:, 1] = xyz[:, 1] * (max_y - min_y) + min_y
        xyz[:, 2] = xyz[:, 2] * (max_z - min_z) + min_z
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)



    try:
        pcd = fetchPly(ply_path, mask=args.use_ground_truth_pose)
        print("Loading point cloud from ", ply_path)
    except:
        pcd = None
        print("No point cloud found")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "ScanNet": readScanNetInfo,
    "Kitti": readKittiInfo,
}
