#!/usr/bin/env bash
# mkdir ./my_model

## Sumission tracking and detection
mode="track"

if [ "$mode" = "track" ];then
    echo "Reading Track Result & Convert Binfile"
    echo "-----------------------------------------------------------------------------------------------------------------------------"
    # track_result='./track_results/All_dla34'
    track_result='/home/member/Workspace/ngocnt/baseline_waymo/waymo_challenge_2020/src/results/dla_34_head_128'
    track_output='./outputs/val_full_pred_fair_newest_128.bin'
    submit_file='val_full_pred_fair_newest_128.tar'

    python waymo_open_dataset/metrics/tools/create_prediction_file_example.py  --track_result_path=${track_result} \
                                                                               --ann_info='/ssd6/data/waymo/val/annotations/val.json' \
                                                                               --output=${track_output}
    echo ""
    echo "Convert Binfile to submission & Compress..."
    bazel-bin/waymo_open_dataset/metrics/tools/create_submission --input_filenames=${track_output} \
                                                                 --output_filename='./my_model_track/model' \
                                                                 --submission_filename='waymo_open_dataset/metrics/tools/submission_track.txtpb'
    tar -cvf ${submit_file} my_model_track
    gzip ${submit_file}

else
    echo "Reading Detection Result & Convert Binfile"
    echo "------------------------------------------------------------------------------------------------------------------------------"
    det_result='./detect_results'
    det_output='./outputs/val_full_pred_fair_newest_det.bin'
    submit_file='val_full_pred_fair_newest_det.tar'

    python waymo_open_dataset/metrics/tools/create_prediction_file_example.py  --track_result_path=${det_result} \
                                                                           --ann_info='/ssd6/data/waymo/val/annotations/val.json' \
                                                                           --output=${det_output} \
                                                                           --mode detect
    echo ""
    echo "Convert Binfile to submission & Compress..."                                                                       
    bazel-bin/waymo_open_dataset/metrics/tools/create_submission --input_filenames=$det_output \
                                                                 --output_filename='./my_model_det/model' \
                                                                 --submission_filename='waymo_open_dataset/metrics/tools/submission_det.txtpb'
    tar -cvf ${submit_file} my_model_det
    gzip ${submit_file}
fi
#
#real data
# bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main ./outputs/val_full_pred.bin \
#                                                                          /home/member/Workspace/dataset/waymo/val/val_gt.bin