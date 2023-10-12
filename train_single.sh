if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_id>"
    exit 1
fi

DIR=`realpath $1`
COLOR_DIR="color"

mamba activate gaussian_splatting
CUDA_VISIBLE_DEVICES=0 python train.py -s $DIR \
    -m $DIR \
    -i $DIR/$COLOR_DIR \
    --data_device cpu \
    --eval \
    --position_lr_init 0.00016 \
    --position_lr_final 0.00016 \
    --densify_from_iter 100 \
    --densify_until_iter 30000 \
    --densify_grad_threshold 0.0002 \
    --no_shuffle_train \
    --lambda_dssim 0.2 \
    --random_init_points \
    --n_random_points 100000 \
    # --use_ground_truth_pose  \
    # --pose_path $DIR/pose 


