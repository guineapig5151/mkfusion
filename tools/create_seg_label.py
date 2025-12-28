import pickle
import copy
import numpy as np
from pcdet.utils import common_utils, box_utils, calibration_kitti
from pathlib import Path
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import os
from utils_seg import COLOR_MAP

def include_vod_data(info_path):
    vod_infos_all = []
    vod_infos = []
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
        if len(infos) == 2:
            vod_infos.extend(infos['data_list'])
        else:
            vod_infos.extend(infos)

    vod_infos_all.extend(vod_infos)
    print(len(vod_infos))
    return vod_infos_all

def get_radar(idx, name):
    global root_split_path
    lidar_file = root_split_path / name / ('%s.bin' % idx)
    assert lidar_file.exists()
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 7)

def get_calib(idx):
    # /data/vod/training/calib/xxxxxx.txt
    global root_split_path
    calib_file = root_split_path / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

color_map = COLOR_MAP
color_map_array = np.array([color_map[i] for i in range(len(color_map))])

train_path = Path('data/vod/kitti_infos_val.pkl')
name = '.'

root_split_path = Path('data/vod_lidar_in_radarcoord')
save_path = 'data/seg_label_lidar/'
if os.path.exists(save_path) == False:
    os.makedirs(save_path)

vod_infos_all = include_vod_data(train_path)
all_frame_ids = [item['point_cloud']['lidar_idx'] for item in vod_infos_all]

class_all = ['Car', 'Pedestrian', 'Cyclist', 
             'bicycle', 'bicycle_rack', 'moped_scooter', 
             'rider', 'human_depiction', 'truck', 
             'ride_other', 'motor', 'vehicle_other', 
             'ride_uncertain']

for index in range(len(all_frame_ids)):
    # index = all_frame_ids.index('02080')

    info = copy.deepcopy(vod_infos_all[index])
    sample_idx = info['point_cloud']['lidar_idx']

    annos = info['annos']
    annos = common_utils.drop_info_with_name(annos, name='DontCare')
    loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
    gt_names = annos['name']
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    calib = get_calib(sample_idx)
    gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
    # for i in range(gt_names.shape[0]):
    #     if gt_names[i] not in class_all:
    #         class_all.append(gt_names[i])
    radar_point = get_radar(sample_idx, name)

    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                radar_point[:, 0:3], gt_boxes_lidar)

    point_label = -1 * np.ones((radar_point.shape[0],))

    vis = False
    for box_id in range(gt_names.shape[0]):
        if gt_names[box_id] == 'moped_scooter':
            vis == True
        # if gt_names[box_id] != 'rider':
        point_label[point_indices[box_id] > 0] = class_all.index(gt_names[box_id])
    
    point_label = point_label.astype(np.int64)

    if vis:
        point_co = color_map_array[point_label + 1]
        point_color = np.concatenate((radar_point[:,0:3], point_co),axis=1)
        
        point_color.astype(np.float32).tofile(save_path + sample_idx + '_color.bin')
    point_label.tofile(save_path + sample_idx + '.bin')
    vis = False

print('end')