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
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def readImages(renders_dir, gt_dir):
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)

        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :]
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :]
        yield render_image, gt_image, fname


def evaluate(source_path, model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = source_path
    try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(source_path) / "test"
        print(test_dir)

        for method in os.listdir(test_dir):
            print("Method:", method)
            if not os.path.isdir(os.path.join(test_dir, method)):
                continue

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            n_render = len(os.listdir(renders_dir))
            it = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []
            image_names = []
            for idx in tqdm(range(n_render), desc="Metric evaluation progress"):
                render, gt, image_name = next(it)
                ssims.append(ssim(render.cuda(), gt.cuda()))
                psnrs.append(psnr(render.cuda(), gt.cuda()))
                lpipss.append(lpips(render.cuda(), gt.cuda(), net_type="vgg"))
                image_names.append(image_name)

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                }
            )
            per_view_dict[scene_dir][method].update(
                {
                    "SSIM": {
                        name: ssim
                        for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                    },
                    "PSNR": {
                        name: psnr
                        for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                    },
                    "LPIPS": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                    },
                }
            )

        with open(scene_dir + "/results.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
    except Exception as e:
        print("Unable to compute metrics for model", scene_dir)
        print("Error:", e)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument("--source_path", "-s", required=True, type=str)
    args = parser.parse_args()
    evaluate(args.source_path, args.model_paths)
