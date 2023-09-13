CUR=$(pwd)
x=8787
conda activate gaussian_splatting
for d in /scratch/data/ScanNet/scans/*; do
    if [ -d $d ]; then
        echo $d | tee -a ScanNet.log
        python train.py -s $d -m $d -i $d/extract --ip 0.0.0.0 --port $x --eval 2>&1 | tee $d/train.log
        python render.py -m $d | tee $d/render.log
        python metrics.py -m $d | tee -a ScanNet.log
	((x++))
    fi
done
