# !/bin/bash

cfg_file=$1

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1"  \
torchrun  \
    --nproc_per_node=2  \
    --nnodes=1  \
    --node_rank=0  \
    --master_addr="localhost"  \
    --master_port=12345  \
    scripts/dist_train.py  -c $cfg_file