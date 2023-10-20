if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_id>"
    exit 1
fi

DIR=`realpath $1`
COLOR_DIR="color_full"
which python

mamba activate /media/jeffyct/Data/mamba/gaussian_splatting
python train.py -s $DIR \
    -m $DIR \
    -i $DIR/$COLOR_DIR \
    --data-device cpu \
    --iterations 10000
    # --eval \
    # --position_lr_init 0.0016 
    # --random_init_points \
    # --n_random_points 100000 \
    # --use_ground_truth_pose  \
    # --pose_path $DIR/pose 


