
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse
import json
from tqdm import tqdm

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# tf.enable_eager_execution()
WAYMO_CLASSES = ['TYPE_UNKNOWN', 'TYPE_VECHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']
CAMERA_NAME = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']

def extract_frame(record_files, outname, outdir_img, class_mapping, dataset_type, resize_ratio=1.0):
    cat_info = []
    for i, cat in enumerate(class_mapping):
        cat_info.append({'name': cat, 'id': i + 1})

    ret = {'images': [], 'annotations': [], "categories": cat_info, 'videos': []}
    bboxes_all = {}
    scores_all = {}
    cls_inds_all = {}
    track_ids_all = {}
    image_names_all = {}
    count = 0
    acc = 0
    
    if "test" in outdir_img or "val" in outdir_img:
        for vid, frames_path in enumerate(tqdm(record_files)):
            # print("Extract record: ", frames_path )
            ret['videos'].append({'id': vid, 'file_name': frames_path})
            id_dict = {}
            dataset = tf.data.TFRecordDataset(frames_path, compression_type='')
            track_id_max = []

            for i, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                
                (range_images, camera_projections, range_image_top_pose) = (
                    frame_utils.parse_range_image_and_camera_projection(frame))
                time = frame.context.stats.time_of_day
                weather = frame.context.stats.weather
                #new
                for i in range(5):
                    im = tf.image.decode_jpeg(frame.images[i].image).numpy()[:,:,::-1]
                    target_size = (int(im.shape[1] * resize_ratio), int(im.shape[0] * resize_ratio))
                    im = cv2.resize(im, target_size)
                    if "val" in outdir_img:
                        labels = frame.camera_labels[i]
                        cv2.imwrite(outdir_img + '/val_{}_record_{}_{}.png'.format(count,vid,open_dataset.CameraName.Name.Name(labels.name)), im)
                        image_info = {'file_path': outdir_img + '/val_{}_record_{}_{}.png'.format(count,vid,open_dataset.CameraName.Name.Name(labels.name)),
                                    'context_name': frames_path.split("/")[-1].split("-")[-1].replace("_with_camera_labels.tfrecord",""),
                                    'camera_type': open_dataset.CameraName.Name.Name(labels.name),
                                    'timestamp_micros': frame.timestamp_micros,
                                    'frame_id': count}
                        count += 1
                    else:
                        image_info = {'file_path': outdir_img + '/test_{}_record_{}_{}.png'.format(count,vid,open_dataset.CameraName.Name.Name(i+1)),
                                    'context_name': frames_path.split("/")[-1].split("-")[-1].replace("_with_camera_labels.tfrecord",""),
                                    'camera_type': open_dataset.CameraName.Name.Name(i+1),
                                    'timestamp_micros': frame.timestamp_micros,
                                    'frame_id': count}
                        count += 1
                    ret['images'].append(image_info)
        json.dump(ret, open(outname.replace(".txt",".json"), 'w'))

        print("Finish convert {} record files, output {} image files".format(vid, count))
    else:
        # for vid, frames_path in enumerate(record_files[0:2]):
        id_dict = {'1':{}, '2':{}, '3':{}, '4':{}, '5':{}}
        for vid, frames_path in enumerate(tqdm(record_files)):
            # print("Extract record: ", frames_path )
            ret['videos'].append({'id': vid, 'file_name': frames_path})
            dataset = tf.data.TFRecordDataset(frames_path, compression_type='')
            # if vid == 1:
                # print(len(track_ids_all))
                # break
                # import ipdb; ipdb.set_trace()
            for i, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                
                (range_images, camera_projections, range_image_top_pose) = (
                    frame_utils.parse_range_image_and_camera_projection(frame))
                time = frame.context.stats.time_of_day
                weather = frame.context.stats.weather
                ##new for train
                for i in range(5):
                    im = tf.image.decode_jpeg(frame.images[i].image).numpy()[:,:,::-1]
                    target_size = (int(im.shape[1] * resize_ratio), int(im.shape[0] * resize_ratio))
                    im = cv2.resize(im, target_size)
                    labels = frame.camera_labels
                    if len(labels) == 0:
                        labels = frame.projected_lidar_labels
                    if len(labels) == 0:
                        break
                    labels = labels[i]

                    # iii = []
                    # print(open_dataset.CameraName.Name.Name(labels.name))
                    # for label in labels.labels:
                    #     iii.append(label.id)
                    # print(iii)

                    boxes, types, ids = extract_labels(labels)
                    # if i == 1:
                        # import ipdb; ipdb.set_trace()
                    bboxes, cls_inds, track_ids = convert_infos(boxes, types, ids, id_dict[str(labels.name)], dataset_type)
                    bboxes *= resize_ratio
                    scores = np.zeros(cls_inds.shape, dtype='f')
                    bboxes_all[count] = bboxes
                    scores_all[count] = scores
                    cls_inds_all[count] = cls_inds
                    track_ids_all[count] = track_ids

                    context_name = frames_path.split("/")[-1].split("-")[-1].replace("_with_camera_labels.tfrecord","")
                    cv2.imwrite(outdir_img + '/train_{}_record_{}_{}_{}.png'.format(count,vid,context_name,open_dataset.CameraName.Name.Name(labels.name)), im)
                    image_info = {'file_path': outdir_img + '/train_{}_record_{}_{}_{}.png'.format(count,vid,context_name,open_dataset.CameraName.Name.Name(labels.name)),
                                'file_name':'train_{}_record_{}_{}_{}.png'.format(count,vid,context_name,open_dataset.CameraName.Name.Name(labels.name)),
                                'id': count,
                                'context_name': context_name,
                                'camera_type': open_dataset.CameraName.Name.Name(labels.name),
                                'timestamp_micros': frame.timestamp_micros,
                                'frame_id': count}
                    count += 1

                    ret['images'].append(image_info)        

        if len(bboxes_all) > 0:
            if 'coco' in dataset_type:
                writeCOCO(outname.replace(".txt",".json"), bboxes_all, scores_all, cls_inds_all, ret, track_ids_all, class_mapping)
        print("Finish convert {} record files, output {} image files".format(vid, count))
        
    ##        old for train 
    #         im = tf.image.decode_jpeg(frame.images[0].image).numpy()[:,:,::-1]
    #         target_size = (int(im.shape[1] * resize_ratio), int(im.shape[0] * resize_ratio))
    #         im = cv2.resize(im, target_size)
    #         labels = frame.camera_labels
    #         if len(labels) == 0:
    #             labels = frame.projected_lidar_labels
    #         if len(labels) == 0:
    #             break
    #         assert labels[0].name == 1
    #         boxes, types, ids = extract_labels(labels[0])
    #         bboxes, cls_inds, track_ids = convert_infos(boxes, types, ids, id_dict, dataset_type)
    #         bboxes *= resize_ratio
    #         scores = np.zeros(cls_inds.shape, dtype='f')
    #         bboxes_all[count] = bboxes
    #         scores_all[count] = scores
    #         cls_inds_all[count] = cls_inds
    #         track_ids = track_ids + acc
    #         track_ids_all[count] = track_ids
    #         if len(track_ids) > 0:
    #             track_id_max.append(np.max(track_ids))
    #         image_names_all[count] = outdir_img + '/%04d.png'%count
    #         # cv2.imwrite(outdir_img + '/%04d.png'%count, im)
    #         image_info = {'file_name': outdir_img + '/%04d.png'%count,
    #                     'id': count,
    #                     'calib': None,
    #                     'video_id': vid ,
    #                     'frame_id': count}
    #         ret['images'].append(image_info)
    #         count += 1

    #     acc = np.max(np.stack(track_id_max)) + 1

    # if len(bboxes_all) > 0:
    #     if 'coco' in dataset_type:
    #         writeCOCO(outname.replace(".txt",".json"), bboxes_all, scores_all, cls_inds_all, ret, track_ids_all, class_mapping)
    #     elif 'kitti' in dataset_type:
    #         writeKITTI(outname, bboxes_all, scores_all, cls_inds_all, track_ids_all, class_mapping)
    #     else:
    #         raise ValueError('could not find dataset type')            

def extract_labels(camera_label):
    box_labels = camera_label.labels
    boxes = []
    types = []
    ids = []
    for box_label in box_labels:
        boxes.append([box_label.box.center_x, box_label.box.center_y, box_label.box.length, box_label.box.width])
        types.append(box_label.type)
        ids.append(box_label.id)
    return boxes, types, ids

def convert_infos(boxes, types, ids, id_dict, dataset_type):
    max_id = max(id_dict.values()) + 1 if len(id_dict) > 0 else 0
    boxes = np.array(boxes)
    if len(boxes) > 0:
        if 'coco' in dataset_type: #COCO Bounding box: (x-top left, y-top left, width, height)
            bboxes = np.zeros_like(boxes)
            bboxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            bboxes[:, 2:] = boxes[:, 2:]
        elif 'kitti' in dataset_type:
            bboxes = np.zeros_like(boxes)
            bboxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            bboxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
        else:
            bboxes = boxes
    else:
        bboxes = np.zeros((0,4), dtype='f')
    
    cls_inds = []
    track_ids = []
    for cls, old_id in zip(types, ids):
        if old_id in id_dict:
            track_id = id_dict[old_id]
        else:
            id_dict[old_id] = max_id
            track_id = max_id
            max_id += 1
        cls_inds.append(cls)
        track_ids.append(track_id)
    cls_inds = np.array(cls_inds)
    track_ids = np.array(track_ids)
    return bboxes, cls_inds, track_ids

def writeKITTI(filename, bboxes, scores, cls_inds, track_ids=None, classes=None):
    f = open(filename, 'w')
    for fid in bboxes:
        for bid in range(len(bboxes[fid])):
            fields = [''] * 17
            fields[0] = fid
            fields[1] = -1 if track_ids is None else int(track_ids[fid][bid])
            fields[2] = classes[int(cls_inds[fid][bid])]
            fields[3:6] = [-1] * 3
            fields[6:10] = bboxes[fid][bid]
            fields[10:16] = [-1] * 6
            fields[16] = scores[fid][bid]
            fields = map(str, fields)
            f.write(' '.join(fields) + '\n')
    f.close()

def writeCOCO(filename, bboxes, scores, cls_inds, ret, track_ids=None, classes=None):
    for fid in bboxes:
        for bid in range(len(bboxes[fid])):
            ann = {'image_id': fid,
                'id': int(len(ret['annotations'])),
                'category_id': cls_inds[fid][bid].tolist(),
                'dim': 3,
                'bbox': bboxes[fid][bid].tolist(),
                #    'depth': location[2],
                #    'alpha': alpha,
                #    'truncated': truncated,
                #    'occluded': occluded,
                #    'location': location,
                #    'rotation_y': rotation_y,
                #    'amodel_center': amodel_center,
                'track_id': track_ids[fid][bid].tolist()}
            ret['annotations'].append(ann)
    json.dump(ret, open(filename, 'w'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record_dir')
    parser.add_argument('output_id')
    parser.add_argument('--workdir', default='.')
    parser.add_argument('--dataset_type', default='coco',
                    choices=['coco', 'kitti'],)
    parser.add_argument('--resize', default=0.5625, type=float)
    args = parser.parse_args()

    os.chdir(args.workdir)
    # if not os.path.exists('images'):
        # os.mkdir('images')
    image_path = os.path.join(args.output_id, "images")
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    label_path = os.path.join(args.output_id, "annotations")
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    record_files = sorted(glob.glob("{}/*/*.tfrecord".format(args.record_dir)))
    # record_files = glob.glob("{}/*.tfrecord".format(args.record_dir))

    # import ipdb; ipdb.set_trace()
    output_ann = os.path.join(label_path + '/{}.txt'.format(args.record_dir.split("/")[-2]))
    extract_frame(record_files, output_ann, image_path, WAYMO_CLASSES, args.dataset_type, resize_ratio=args.resize)

if __name__ == "__main__":
    main()
