# run file python preprocess.py --data_root data/SHREC16_test/cuts --no_normalize --n_eig 200

# echo "Preprocessing SHREC16_test/holes"
# python preprocess.py --data_root data/SHREC16_test/holes --no_normalize --n_eig 200

# echo "Preprocessing SHREC16_test/null"
# python preprocess.py --data_root data/SHREC16_test/null --no_normalize --n_eig 200

# echo "Preprocessing SMAL_r"
# python preprocess.py --data_root data/SMAL_r --no_normalize --n_eig 200

echo "Preprocessing TOPKIDS"
python preprocess.py --data_root data/TOPKIDS --no_normalize --n_eig 200