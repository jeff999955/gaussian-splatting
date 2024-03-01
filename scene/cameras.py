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

import numpy as np
import torch
from PIL import Image
from torch import nn

from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,  # in world view, i.e. from w2c
        T,  # in world view, i.e. from w2c
        FoVx,
        FoVy,
        image_path,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        fx=None,
        fy=None,
        cx=None,
        cy=None,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_path = image_path
        self.image_name = image_name
        self.image_height = None
        self.image_width = None

        try:
            with Image.open(self.image_path) as image:
                self.image_width, self.image_height = image.size
        except:
            self.image_width, self.image_height = 1408, 376

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")
        
        self.fx = fx or fov2focal(FoVx, self.image_width)
        self.fy = fy or fov2focal(FoVy, self.image_height)

        self.cx = cx or self.image_width / 2
        self.cy = cy or self.image_height / 2

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                self.znear, self.zfar, self.FoVx, self.FoVy, self.cx, self.cy, self.image_width, self.image_height 
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def original_image(self):
        image = Image.open(self.image_path)

        target_size = (
            int(self.image_width * self.scale),
            int(self.image_height * self.scale),
        )
        image_tensor = PILtoTorch(image, target_size)
        image.close()
        image_tensor = image_tensor.clamp(0.0, 1.0).to(self.data_device)

        alpha_mask = None
        if image_tensor.shape[1] == 4:
            alpha_mask = image_tensor[3:4, ...]

        if alpha_mask:
            image_tensor *= alpha_mask.to(self.data_device)
        else:
            image_tensor *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )

        return image_tensor


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
