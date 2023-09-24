CUR=$(pwd)
x=8787
conda activate gaussian_splatting
for d in /scratch/data/ScanNet/scans/*; do
    if [ -d $d ]; then
        zip -ur scannet.zip $d/test/ours_30000/renders/*.png
    fi
done
