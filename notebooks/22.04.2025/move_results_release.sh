base_dir=/home/s94zalek_hpc/DenoisingFunctionalMaps/results/ddpm_64_SMAL
# for each subdir in $base_dir/*; do

for subdir in $base_dir/*; do
    echo "Processing $subdir"
    # Check if the directory contains a file named "results.txt"
    mkdir $subdir/p2p
    mv $subdir/*.pt $subdir/p2p
done