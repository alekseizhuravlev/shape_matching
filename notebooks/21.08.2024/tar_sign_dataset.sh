#!/bin/bash
echo "Tarring the sign dataset"
base_dir='/home/s94zalek_hpc/shape_matching/data_sign_training/legacy'
# for each folder in the base directory, except FAUST_a and SURREAL, tar the folder, without the global path
for folder in $(ls $base_dir)
do
    if [ $folder == "FAUST_a" ] || [ $folder == "SURREAL" ]
    then
        echo "Skipping $folder"
        continue
    fi
    echo "Tarring $folder"
    tar -cf $base_dir/$folder.tar -C $base_dir $folder
    # echo "Removing $folder"
    # rm -r $base_dir/$folder
done

# dest_dir='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_sign_training/train'
# # move all tar files to the destination directory
# for tar_file in $(ls $base_dir)
# do
#     echo "Moving $tar_file"
#     mv $base_dir/$tar_file $dest_dir
# done