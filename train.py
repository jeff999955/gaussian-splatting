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
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import torchvision
from icecream import ic
from tqdm import tqdm

import wandb
from arguments import argparser
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

LOG = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    shuffle_train=True,
):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset, gaussians, shuffle=shuffle_train, resolution_scales=[args.resolution]
    )

    n_train_cams = len(scene.getTrainCameras(scale=args.resolution))

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if LOG:
        ic(gaussians.get_xyz.shape)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale=args.resolution).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if LOG:
            ic(np.sum(np.isnan(gaussians.get_xyz.detach().cpu().numpy())))
            ic(np.prod(gaussians.get_xyz.shape))

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                iteration,
                n_train_cams,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                is_kitti_test=args.is_kitti_test,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def training_report(
    iteration,
    n_train_cams,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    is_kitti_test=False,
):
    is_render_gt = not is_kitti_test
    if LOG:
        wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "iteration": iteration})
        wandb.log(
            {"train_loss_patches/total_loss": loss.item(), "iteration": iteration}
        )
        wandb.log({"iter_time": elapsed, "iteration": iteration})

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras(scale=args.resolution)},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras(scale=args.resolution)[idx % n_train_cams]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                render_path = os.path.join(
                    scene.model_path,
                    config["name"],
                    "ours_{}".format(iteration),
                    "renders",
                )

                gts_path = os.path.join(
                    scene.model_path,
                    config["name"],
                    "ours_{}".format(iteration),
                    "gt",
                )

                os.makedirs(render_path, exist_ok=True)
                os.makedirs(gts_path, exist_ok=True)

                print("Saving renders and gt images to {}".format(scene.model_path))

                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.model, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    torchvision.utils.save_image(
                        image, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
                    )
                    if is_render_gt:
                        gt_image = torch.clamp(
                            viewpoint.original_image.to("cuda"), 0.0, 1.0
                        )
                        torchvision.utils.save_image(
                            gt_image, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
                        )


                    # TODO: Find out how to add image to wandb
                    # if idx < 5:
                    #     tb_writer.add_images(
                    #         config["name"]
                    #         + "_view_{}/render".format(viewpoint.image_name),
                    #         image[None],
                    #         global_step=iteration,
                    #     )
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(
                    #             config["name"]
                    #             + "_view_{}/ground_truth".format(viewpoint.image_name),
                    #             gt_image[None],
                    #             global_step=iteration,
                    #         )
                    if is_render_gt:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                if is_render_gt:
                    psnr_test /= len(config["cameras"])
                    l1_test /= len(config["cameras"])
                    print(
                        "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                            iteration, config["name"], l1_test, psnr_test
                        )
                    )
                else:
                    print("Specified not to render gt images")
                if LOG:
                    wandb.log(
                        {
                            config["name"] + "/loss_viewpoint - l1_loss": l1_test,
                            "iteration": iteration,
                        }
                    )
                    wandb.log(
                        {
                            config["name"] + "/loss_viewpoint - psnr": psnr_test,
                            "iteration": iteration,
                        }
                    )

        opacity_list = scene.model.get_opacity.detach().cpu().numpy()
        min_opacity = np.min(opacity_list)
        max_opacity = np.max(opacity_list)
        histogram = np.histogram(
            opacity_list, bins=100, range=(min_opacity, max_opacity)
        )
        if LOG:
            wandb.log(
                {
                    "opacity_histogram": wandb.Histogram(np_histogram=histogram),
                }
            )
            wandb.log(
                {"total_points": scene.model.get_xyz.shape[0], "iteration": iteration}
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    LOG = os.getenv("LOG") or False
    parser = argparser(desc="Training script parameters")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[i * 5000 for i in range(20)],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[i * 5000 for i in range(20)],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--no_shuffle_train", action="store_true")
    parser.add_argument("--random_init_points", action="store_true")
    parser.add_argument("--is_kitti_test", action="store_true", default=False)
    parser.add_argument("--random_init_pcd", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    ic(vars(args))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    os.makedirs(args.model_path, exist_ok=True)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if LOG:
        wandb.init(
            # set the wandb project where this run will be logged
            project="gaussian-splatting",
            # track hyperparameters and run metadata
            config=vars(args),
        )
    training(
        args,
        args,
        args,
        args.test_iterations,
        args.save_iterations,
        args.save_iterations,
        args.start_checkpoint,
        args.debug_from,
        not args.no_shuffle_train,
    )

    # All done
    print("\nTraining complete.")
    if LOG:
        wandb.finish()
