# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
"""A simple example to generate a file that contains serialized Objects proto."""

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
import argparse
import struct
import json
import glob
from tqdm import tqdm
from time import time

## Origin WH 1920x1280, 1920x886
Center_RATIO = {'FRONT':0.5625, 'FRONT_LEFT':0.5625, 'SIDE_LEFT':(0.5625,0.5625), 'FRONT_RIGHT':0.5625, 'SIDE_RIGHT':(0.5625,0.5625)}  #HW/hw (1080x720)
Fair_RATIO = {'FRONT':0.5625, 'FRONT_LEFT':0.5625, 'SIDE_LEFT':(0.5625,0.5625), 'FRONT_RIGHT':0.5625, 'SIDE_RIGHT':(0.5625,0.5625)}  #HW/hw (1080x720, 1080x498)
RATIO = 0.5625

CAT_TYPES = [
	label_pb2.Label.TYPE_VEHICLE,
	label_pb2.Label.TYPE_PEDESTRIAN,
	label_pb2.Label.TYPE_CYCLIST,
]
CAM_TYPES = {
	'FRONT': dataset_pb2.CameraName.FRONT,
	'FRONT_LEFT': dataset_pb2.CameraName.FRONT_LEFT,
	'FRONT_RIGHT': dataset_pb2.CameraName.FRONT_RIGHT,
	'SIDE_LEFT': dataset_pb2.CameraName.SIDE_LEFT,
	'SIDE_RIGHT': dataset_pb2.CameraName.SIDE_RIGHT,
}


def _create_pd_obj(frame, res, RATIO, mode):
	"""Creates a prediction objects file."""
	objs = []
	# This is only needed for 2D detection or tracking tasks.
	# Set it to the camera name the prediction is for.
	if len(res) > 0:
		for re in res:
			# import ipdb; ipdb.set_trace()
			re = re.split(",")
			obj = metrics_pb2.Object()
			obj.context_name = frame['context_name']
			invalid_ts = -1
			obj.frame_timestamp_micros =  frame['timestamp_micros']
			obj.camera_name = CAM_TYPES[frame['camera_type']]
			# Populating box and score.
			cx,cy,w,h = [float(x) for x in re[3:7]]
			if mode == 'track':
				track_id = "{}_{}".format(frame['camera_type'],re[1])
			else:
				track_id = str(0)
			if isinstance(RATIO,tuple):
				cx,cy,w,h = cx/RATIO[0], cy/RATIO[1], w/RATIO[0], h/RATIO[1]
			else:
				cx,cy,w,h = cx/RATIO, cy/RATIO, w/RATIO, h/RATIO

			box = label_pb2.Label.Box()
			box.center_x = cx
			box.center_y = cy
			box.center_z = 0
			box.length = w
			box.width = h
			box.height = 0
			box.heading = 0

			obj.object.box.CopyFrom(box)
			# This must be within [0.0, 1.0]. It is better to filter those boxes with
			# small scores to speed up metrics computation.
			obj.score = float(re[-2])
			# For tracking, this must be set and it must be unique for each tracked
			# sequence.
			obj.object.id = track_id #'unique object tracking ID'
			# Use correct type.
			obj.object.type = CAT_TYPES[int(float(re[2]))]
			objs.append(obj)
	else:
		obj = metrics_pb2.Object()
		obj.context_name = (frame['context_name'])
		invalid_ts = -1
		obj.frame_timestamp_micros =  frame['timestamp_micros']
		obj.camera_name = CAM_TYPES[frame['camera_type']]
		# Populating box and score.
		box = label_pb2.Label.Box()
		box.center_x = 0
		box.center_y = 0
		box.center_z = 0
		box.length = 0
		box.width = 0
		box.height = 0
		box.heading = 0

		obj.object.box.CopyFrom(box)
		# This must be within [0.0, 1.0]. It is better to filter those boxes with
		# small scores to speed up metrics computation.
		obj.score = float(0.0)
		# For tracking, this must be set and it must be unique for each tracked
		# sequence.
		obj.object.id = str(0) #'unique object tracking ID'
		# Use correct type.
		obj.object.type = int(0)
		# Add more objects. Note that a reasonable detector should limit its maximum
		# number of boxes predicted per frame. A reasonable value is around 400. A
		# huge number of boxes can slow down metrics computation.

		# Write objects to a file.
		objs.append(obj)

	return objs


def main():
	start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--track_result_path')
	parser.add_argument('--ann_info')
	parser.add_argument('--output')
	parser.add_argument('--mode', default='track', help='track | detect')
	args = parser.parse_args()
	print("Start convert result prediction to binary format!")
	with open(args.ann_info) as f: 
		infos = json.load(f)
	## for tracking output
	if args.mode == 'track':
		results = {'FRONT':{}, 'FRONT_LEFT':{}, 'SIDE_LEFT':{}, 'FRONT_RIGHT':{}, 'SIDE_RIGHT':{}}
		counts = {'FRONT':0, 'FRONT_LEFT':0, 'SIDE_LEFT':0, 'FRONT_RIGHT':0, 'SIDE_RIGHT':0}

		result_files = glob.glob("{}/*.txt".format(args.track_result_path))
		assert len(result_files) == 5 # just 5 camera directions corresponding with 5 results
		objects = metrics_pb2.Objects()
		for i,res in enumerate(result_files):
			camera_type = res.split("/")[-1].split("-")[-1].replace(".txt","")
			print(camera_type)
			with open(res,'r') as r:
				results[camera_type] = r.readlines()
				print("Load result files: {}".format(res))

		for i, frame in enumerate(tqdm(infos['images'])):
			camera_type = frame['camera_type']
			timestamp = frame['timestamp_micros']
			# res = [line for line in results[camera_type] if line.split(",")[-1].replace("\n","") == str(timestamp)]
			res = []
			start = counts[camera_type]
			for line in results[camera_type][start:-1]:
				if line.split(",")[-1].replace("\n","") == str(timestamp):
					res.append(line)
					counts[camera_type]+=1
				else:
					break
					
			if "center" in args.output:
				objs = _create_pd_obj(frame, res, Center_RATIO[camera_type], args.mode)
			elif "fair" in args.output:
				objs = _create_pd_obj(frame, res, Fair_RATIO[camera_type], args.mode)
			else:
				raise ValueError('Add method type to output file')
			for o in objs:
				objects.objects.append(o)

	## for detection output
	else:
		objects = metrics_pb2.Objects()
		result_files = "{}/val_detect.txt".format(args.track_result_path)
		with open(result_files) as r:
			results = r.readlines()
			print("Load result files: {}".format(result_files))
		count = 0
		for i, frame in enumerate(tqdm(infos['images'])):
			timestamp = frame['timestamp_micros']
			res = []
			for line in results[count:-1]:
				if (int(line.split(",")[0]) - 1) == frame['frame_id']:
					res.append(line)
					count+=1
				else:
					break
			objs = _create_pd_obj(frame, res, RATIO, args.mode)
			for o in objs:
				objects.objects.append(o)

	## Gen the *.bin file
	with open(args.output, 'wb') as f:
		f.write(objects.SerializeToString())
	f.close()
	print("Time to create: {} minutes", (time()-start)/60)

if __name__ == '__main__':
	main()



'''
### From ThuyNC

# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
"""A simple example to generate a file that contains serialized Objects proto."""
import numpy as np
import mmcv, argparse, os
from mmdet.datasets import build_dataset
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
_CAT_TYPES = [
	label_pb2.Label.TYPE_VEHICLE,
	label_pb2.Label.TYPE_PEDESTRIAN,
	label_pb2.Label.TYPE_SIGN,
	label_pb2.Label.TYPE_CYCLIST,
]
_CAM_TYPES = {
	'FRONT': dataset_pb2.CameraName.FRONT,
	'FRONT_LEFT': dataset_pb2.CameraName.FRONT_LEFT,
	'FRONT_RIGHT': dataset_pb2.CameraName.FRONT_RIGHT,
	'SIDE_LEFT': dataset_pb2.CameraName.SIDE_LEFT,
	'SIDE_RIGHT': dataset_pb2.CameraName.SIDE_RIGHT,
}
def convert_single_frame(det_result, filename, timestamp_micros, is_gt=False, thres=0.0):
	# Get context name
	context_name = filename.split('-')[-1].split('.')[0]
	context_name = context_name.replace('_with_camera_labels', '')
	# Get camera name
	eles = filename.split('_')
	if "FRONT" in eles[-1]:
		camera_type = filename.split('_')[-1].split('.')[0]
	else:
		camera_type = '_'.join(filename.split('_')[-2:]).split('.')[0]
	camera_name = _CAM_TYPES[camera_type]
	# Get bboxes
	bboxes = []
	if not is_gt:
		for cat_idx, cat_bboxes in enumerate(det_result):
			cat_type = _CAT_TYPES[cat_idx]
			for bbox in cat_bboxes:
				bboxes.append((cat_type, bbox))
	else:
		for bbox, label in zip(*det_result):
			cat_type = _CAT_TYPES[label-1]
			bbox = np.array(bbox.tolist()+[1.0])
			bboxes.append((cat_type, bbox))
	# Create pb2 objects
	pb2_objects = []
	num_objects = 0
	for cat_type, bbox in bboxes:
		x1, y1, x2, y2, sc = bbox.tolist()
		if sc < thres:
			continue
		cx = (x1 + x2) / 2
		cy = (y1 + y2) / 2
		w = x2 - x1
		h = y2 - y1
		num_objects += 1
		o = metrics_pb2.Object()
		# The following 3 fields are used to uniquely identify a frame a prediction
		# is predicted at. Make sure you set them to values exactly the same as what
		# we provided in the raw data.
		o.context_name = (context_name)
		# The frame timestamp for the prediction. See Frame::timestamp_micros in
		# dataset.proto.
		invalid_ts = -1
		o.frame_timestamp_micros = timestamp_micros
		# This is only needed for 2D detection or tracking tasks.
		# Set it to the camera name the prediction is for.
		o.camera_name = camera_name
		# Populating box and score.
		box = label_pb2.Label.Box()
		box.center_x = cx
		box.center_y = cy
		box.center_z = 0
		box.length = w
		box.width = h
		box.height = 0
		box.heading = 0
		o.object.box.CopyFrom(box)
		# This must be within [0.0, 1.0]. It is better to filter those boxes with
		# small scores to speed up metrics computation.
		o.score = sc
		# For tracking, this must be set and it must be unique for each tracked
		# sequence.
		# o.object.id = 'unique object tracking ID'
		# Use correct type.
		o.object.type = cat_type
		pb2_objects.append(o)
	# Return
	print("context_name: {}; camera_type: {}; timestamp_micros: {}; Bboxes: {}".format(
		filename, camera_type, timestamp_micros, num_objects))
	return pb2_objects
if __name__ == '__main__':
	# ArgumentParser
	parser = argparse.ArgumentParser(description='ArgumentParser')
	parser.add_argument('--pkl', help='Prediction-result pickle file')
	parser.add_argument('--out', help='Output bin file for submitting')
	parser.add_argument('--cfg', help='Config file', default="configs/waymo/val_dataset.py")
	parser.add_argument('--gt', action='store_true', default=False, help='Use gt for submitting')
	parser.add_argument('--thres', type=float, default=0.1, help='Score threshold')
	args = parser.parse_args()
	# Build dataset
	cfg = mmcv.Config.fromfile(args.cfg)
	dataset = build_dataset(cfg.data.test)
	print("Dataset size:", len(dataset))
	# Load pkl result
	det_results = mmcv.load(args.pkl)
	# Use gt
	if args.gt:
		det_results = []
		for idx in range(len(dataset)):
			ann = dataset.get_ann_info(idx)
			bboxes = ann['bboxes']
			labels = ann['labels'] # start-from-1
			det_results.append((bboxes, labels))
	else:
		assert len(dataset) == len(det_results), "len(dataset)={}; len(det_results)={}".format(
			len(dataset), len(det_results))
	# Convert for each frames
	objects = metrics_pb2.Objects()
	for idx, det_result in enumerate(det_results):
		filename = dataset.img_infos[idx]['filename']
		timestamp_micros = dataset.img_infos[idx]['timestamp_micros']
		pb2_objects = convert_single_frame(
			det_result, filename, timestamp_micros, args.gt, args.thres)
		for o in pb2_objects:
			objects.objects.append(o)
	# Write output
	dirname = os.path.dirname(args.out)
	os.makedirs(dirname, exist_ok=True)
	with open(args.out, 'wb') as fp:
		fp.write(objects.SerializeToString())
	print("Result .bin file is saved at {}".format(args.out))'''