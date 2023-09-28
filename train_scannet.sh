CUR=$(pwd)
x=8787
conda activate gaussian_splatting

dir=(/scratch/data/ScanNet_sparse/scans/scene0101_00 /scratch/data/ScanNet_sparse/scans/scene0241_00)
for d in ${dir[@]}; do
    if [ -d $d ]; then
        echo $d | tee -a ScanNet.log
        python train.py -s $d -m $d -i $d/color --data_device cpu --eval 2>&1 | tee $d/train.log
        python render.py -m $d | tee $d/render.log
        python metrics.py -m $d | tee -a ScanNet.log
	((x++))
    fi
done
