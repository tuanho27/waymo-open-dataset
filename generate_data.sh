#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3;

# python convert_tfrecord.py /ssd6/data/waymo/val_records/ /ssd6/data/waymo/val 
# python convert_tfrecord.py /ssd6/data/waymo/test_records/ /ssd6/data/waymo/test
python convert_tfrecord.py /ssd6/data/waymo/train_records/ /ssd6/data/waymo/train

# python convert_tfrecord.py /ssd6/data/waymo/val_records/validation_0000/ /home/tuanho/Workspace/waymo/FairMOT/src/test
# python convert_tfrecord.py /ssd6/data/waymo/train_records/training_0000/ /home/member/Workspace/dataset/waymo/train/example/0000
# python convert_tfrecord.py /ssd6/data/waymo/test_records/testing_0000/ /home/member/Workspace/dataset/waymo/test/0000
