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
import os

import torch
import tqdm
from PIL import Image


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def resize_all(path):
    imgs = glob.glob(os.path.join(path, "*.jpg"))
    for img in tqdm.tqdm(imgs):
        with Image.open(img) as image:
            image = image.resize((640, 480))
        image.save(img)


if __name__ == "__main__":
    import sys

    resize_all(sys.argv[1])
