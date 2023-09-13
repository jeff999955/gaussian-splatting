CUR=$(pwd)
x=8787
conda activate gaussian_splatting
for d in /scratch/data/Replica/*; do
    if [ -d $d ]; then
        echo $d
        python train.py -s $d -m $d -i $d/extract --ip 0.0.0.0 --port $x --eval --debug_from 1 --no_init_pcd 2>&1 | tee $d/train_wo_init.log
        # python render.py -m $d
        # python metrics.py -m $d
	((x++))
    break
    fi
done
