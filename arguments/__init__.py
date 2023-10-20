from argparse import ArgumentParser


def include_model_params(parser):
    loading_args = parser.add_argument_group("Model Parameters")
    loading_args.add_argument(
        "--random-init-points",
        default=False,
        action="store_true",
        help="Random initialization points",
    )
    loading_args.add_argument(
        "--n-random-points",
        default=100_000,
        type=int,
        help="Number of random points, only used if random-init-points is True",
    )
    loading_args.add_argument(
        "--use-ground-truth-pose",
        default=False,
        action="store_true",
        help="Use ground truth pose",
    )
    loading_args.add_argument("--pose-path", default="", help="Pose path")
    loading_args.add_argument(
        "--split-setting",
        default="pointnerf",
        choices=["pointnerf", "mipnerf"],
        help="Split setting",
    )
    loading_args.add_argument("--sh-degree", default=3, type=int, help="SH Degree")
    loading_args.add_argument("--source-path", "-s", default="", help="Source path")
    loading_args.add_argument("--model-path", "-m", default="", help="Model path")
    loading_args.add_argument("--images", "-i", default="images", help="Images")
    loading_args.add_argument("--resolution", default=-1, type=int, help="Resolution")
    loading_args.add_argument(
        "--white-background",
        default=False,
        action="store_true",
        help="White background",
    )
    loading_args.add_argument("--data-device", default="cuda", help="Data device")
    loading_args.add_argument(
        "--eval", default=False, action="store_true", help="Evaluation mode"
    )


def include_pipeline_params(parser):
    pipeline_args = parser.add_argument_group("Pipeline Parameters")
    pipeline_args.add_argument("--convert_SHs_python", action="store_true")
    pipeline_args.add_argument("--compute_cov3D_python", action="store_true")
    pipeline_args.add_argument("--debug", action="store_true")


def include_optimization_params(parser):
    optimization_args = parser.add_argument_group("Optimization Parameters")
    optimization_args.add_argument("--iterations", default=30_000, type=int)
    optimization_args.add_argument("--position_lr_init", default=0.00016, type=float)
    optimization_args.add_argument("--position_lr_final", default=0.0000016, type=float)
    optimization_args.add_argument("--position_lr_delay_mult", default=0.01, type=float)
    optimization_args.add_argument("--position_lr_max_steps", default=30_000, type=int)
    optimization_args.add_argument("--feature_lr", default=0.0025, type=float)
    optimization_args.add_argument("--opacity_lr", default=0.05, type=float)
    optimization_args.add_argument("--scaling_lr", default=0.005, type=float)
    optimization_args.add_argument("--rotation_lr", default=0.001, type=float)
    optimization_args.add_argument("--percent_dense", default=0.01, type=float)
    optimization_args.add_argument("--lambda_dssim", default=0.2, type=float)
    optimization_args.add_argument("--densification_interval", default=100, type=int)
    optimization_args.add_argument("--opacity_reset_interval", default=3000, type=int)
    optimization_args.add_argument("--densify_from_iter", default=500, type=int)
    optimization_args.add_argument("--densify_until_iter", default=15_000, type=int)
    optimization_args.add_argument(
        "--densify_grad_threshold", default=0.0002, type=float
    )


def argparser(desc="") -> ArgumentParser:
    parser = ArgumentParser(description=desc)

    include_model_params(parser)
    include_pipeline_params(parser)
    include_optimization_params(parser)

    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    print(args)
