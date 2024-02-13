#!/usr/bin/zsh
# DIR="/jeffyct-ssd/data/nerf_synthetic_colmap"
# MODEL_DIR="./nerf_synthetic_colmap"

# for scene in `ls $DIR`; do
#     echo $scene
#     if [ -d $DIR/$scene ]; then
#         for i in {0..9}; do
#             echo $i
#             mkdir -p $MODEL_DIR/$scene/$stride/$i
#             python train.py -s $DIR/$scene \
#                 -m $MODEL_DIR/$scene/$stride/$i \
#                 --eval \
#                 --seed $((i*100))

#             python render.py -s $DIR/$scene \
#                 -m $MODEL_DIR/$scene/$stride/$i \
#                 --eval  \
#                 --seed $((i*100))
#         done
#     fi
# done

DIR="/jeffyct-ssd/data/nerf_synthetic_colmap"
MODEL_DIR="./nerf_synthetic_stride_colmap"
for stride in 2 4 8; do
    for scene in `ls $DIR`; do
        echo $scene
        if [ -d $DIR/$scene ]; then
            echo scene: $scene, stride: $stride, run: $i
            mkdir -p $MODEL_DIR/$scene/$stride/$i
            is_train=1
            is_render=1
            if [ -d $MODEL_DIR/$scene/$stride/$i/point_cloud ]; then
                echo already found a trained run, skipping train
                is_train=0
            fi
            if [ -d $MODEL_DIR/$scene/$stride/$i/point_cloud ]; then
                echo already found a rendered run, skipping
                is_render=0
            fi
            
            if [ $is_train -ne 0 ]; then
                python train.py -s $DIR/$scene \
                    -m $MODEL_DIR/$scene/$stride/$i \
                    --eval \
                    --seed $((i*100)) \
                    --stride $stride
            fi 
            
            if [ $is_render -ne 0 ]; then
                python render.py -s $DIR/$scene \
                    -m $MODEL_DIR/$scene/$stride/$i \
                    --eval  \
                    --seed $((i*100)) \
                    --stride $stride
            fi
        fi
    done
done
