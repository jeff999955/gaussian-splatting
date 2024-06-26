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

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from icecream import ic
from tqdm import tqdm

from arguments import argparser
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, margin=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    print("Rendering to", render_path)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        output_rendering = torch.zeros_like(rendering)
        output_rendering[:, margin:-margin, margin:-margin] = rendering[:, margin:-margin, margin:-margin]
        output_gt = torch.zeros_like(gt)
        output_gt[:, margin:-margin, margin:-margin] = gt[:, margin:-margin, margin:-margin]
        torchvision.utils.save_image(
            output_rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            output_gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


def render_sets(
    dataset,
    iteration: int,
    pipeline,
    skip_train: bool,
    skip_test: bool,
):
    is_scannet = os.path.exists(os.path.join(args.source_path, "pose"))
    margin = 10 if is_scannet else 0
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        ic(vars(dataset))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                margin=margin
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                margin=margin
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparser(desc="Testing script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ic(args)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        args,
        args.iteration,
        args,
        args.skip_train,
        args.skip_test,
    )
