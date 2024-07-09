#!/bin/bash
echo "job.sh called with the following command line arguments:"
echo $@
source ~/.bashrc

# /home1/09241/jonfrey/perception_bev_learning/scripts/cluster/copy_datasets_to_tmpdir.sh

/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 /home1/09241/jonfrey/perception_bev_learning/scripts/preprocessing/create_compact_h5py_file_new.py $@