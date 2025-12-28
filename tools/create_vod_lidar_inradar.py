
import numpy as np
import calibration_kitti
from pathlib import Path
import pickle
import os
import copy

def get_lidar(idx, name):
    global lidar_path
    lidar_file = lidar_path / name / ('%s.bin' % idx)
    assert lidar_file.exists()
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

def get_radar(idx, name):
    global root_split_path
    lidar_file = root_split_path / name / ('%s.bin' % idx)
    assert lidar_file.exists()
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 7)

def trans_lidar_radar(lidar_point, calib_lidar, calib_radar):
    lidar_point_xyz = lidar_point[:,:3]
    lidar_point_in_rect = calib_lidar.lidar_to_rect(lidar_point_xyz)
    lidar_point_in_radar = calib_radar.rect_to_lidar(lidar_point_in_rect)
    lidar_point_radar = np.concatenate((lidar_point_in_radar, lidar_point[:,3:]), axis = 1)
    return lidar_point_radar

def get_calib_lidar(idx):
    global lidar_path
    calib_file = Path(lidar_path) / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)
        
def get_calib(idx):
    # /data/vod/training/calib/xxxxxx.txt
    global root_split_path
    calib_file = root_split_path / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

def include_vod_data(info_path):
    vod_infos_all = []
    vod_infos = []
    with open(info_path, 'rb') as f:
        # 读取文件为infos
        infos = pickle.load(f) # len():3712
        # 将改内容添加到vod_info列表中
        if len(infos) == 2:
            vod_infos.extend(infos['data_list'])
        else:
            vod_infos.extend(infos)

    # 　变量存储到类中
    vod_infos_all.extend(vod_infos)
    print(len(vod_infos))
    return vod_infos_all

lidar_path = Path('data/vod/lidar/training/')
root_split_path = Path('data/vod/radar_5frames/training/')
name = 'velodyne_reduced'

all_frame_ids = os.listdir('data/vod/lidar/training/velodyne')
all_frame_ids = [item.split('.')[0] for item in all_frame_ids]

save_path = 'data/vod/lidar_in_radarcoord/training/lidar'
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path + '_radar/', exist_ok=True)

for index in range(len(all_frame_ids)):
    sample_idx = all_frame_ids[index]
    lidar_points = get_lidar(sample_idx, name)
    calib_lidar = get_calib_lidar(sample_idx)
    calib = get_calib(sample_idx)
    lidar_in_radar_coord = trans_lidar_radar(lidar_points, calib_lidar, calib)
    
    lidar_color = np.concatenate((lidar_in_radar_coord, np.ones((lidar_in_radar_coord.shape[0], 3))), axis=1)

    lidar_color.astype(np.float32).tofile(save_path + '/' + sample_idx + '.bin')
   
    radar_points = get_radar(sample_idx, name)
    lidar_radar = np.concatenate((radar_points, lidar_color), axis=0)
    lidar_radar.astype(np.float32).tofile(save_path + '_radar/' + sample_idx + '.bin')

