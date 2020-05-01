#!/usr/bin/env bash

# mkdir ./my_model
# python waymo_open_dataset/metrics/tools/create_prediction_file_example.py  --track_result_path='./track_results' \
python waymo_open_dataset/metrics/tools/create_prediction_file_example.py  --track_result_path='/home/tuanho/Workspace/waymo/FairMOT/src/results/All_dla34/' \
                                                                           --ann_info='/ssd6/data/waymo/val/annotations/val.json' \
                                                                           --output='./outputs/val_full_pred_fair.bin' 

# bazel-bin/waymo_open_dataset/metrics/tools/create_submission --input_filenames='./outputs/val_full_pred_fair.bin' \
#                                                              --output_filename='./my_model/model' \
#                                                              --submission_filename='waymo_open_dataset/metrics/tools/submission.txtpb'

# bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main waymo_open_dataset/metrics/tools/fake_predictions.bin  \
                                                                        # waymo_open_dataset/metrics/tools/fake_ground_truths.bin

# bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main waymo_open_dataset/metrics/tools/fake_predictions.bin  \
                                                                        # waymo_open_dataset/metrics/tools/fake_ground_truths.bin

#real data
# bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main ./outputs/val_full_pred.bin \
#                                                                          /home/member/Workspace/dataset/waymo/val/val_gt.bin
