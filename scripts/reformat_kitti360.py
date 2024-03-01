import os
import shutil

import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--logdir", type=str, default="./logdir", help="Directory to move the images."
)
parser.add_argument(
    "--rendered_dir",
    type=str,
    default="./exp/kitti_full",
    help="Directory from which to move the images.",
)
parser.add_argument(
    "--rendered_dir_name",
    type=str,
    default="test_real",
    help="Name for each directory that stores the testing set.",
)
args = parser.parse_args()

logdir = args.logdir

dir_root = "/ubc/cs/research/kmyi/wsun/scratch/dataset/kitti360/"
dir_data_2d_nvs_drop = os.path.join(dir_root, "data_2d_nvs_drop50")

scans = ["test_00", "test_01", "test_02", "test_03", "test_04"]
test_iters = [30000] * len(scans)

save_img_dir = os.path.join(logdir, "test_submit")
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

for scan, test_iter in zip(scans, test_iters):
    fn_frame_list = os.path.join(dir_data_2d_nvs_drop, f"{scan}.txt")
    with open(fn_frame_list) as f:
        frame_00_list = f.read().splitlines()
    image_paths = frame_00_list
    id_list = list(range(len(image_paths)))

    for i in id_list:
        # source
        src_dir = os.path.join(
            args.rendered_dir,
            scan.replace("test_", "train_"),
            args.rendered_dir_name,
            f"ours_{test_iter}",
            "renders"
        )
        filename = f"{i:05d}.png"
        src_img_filename = os.path.join(src_dir, filename)

        image_path = image_paths[i]
        target_img_filename = os.path.basename(image_path)
        seq_name = image_path.split("/")[0].split("_")[-2]
        target_img_filename = seq_name + "_" + target_img_filename
        target_img_filename = os.path.join(save_img_dir, target_img_filename)
        assert os.path.exists(src_img_filename)
        shutil.copyfile(src_img_filename, target_img_filename)
