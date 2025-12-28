import os
import os.path as osp
import cv2
import time
import sys

import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
import random
import numpy as np

from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data

con = 'models/segmentors/mono/configs/HourglassDecoder/vit.raft5.giant2.py'
load = 'metric3d_v2/weight/metric_depth_vit_giant2_800k.pth'
test_data_path = 'data/vod/training/image_2/'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default=con, help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', default=load, help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', default='None', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], help='job launcher')
    parser.add_argument('--test_data_path', default=test_data_path, type=str, help='the path of test data')
    args = parser.parse_args()
    return args

def get_depth_feat_in_model(img=None, input_metas=None, model=None, cfg=None):
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)

    cfg.distributed = False

    test_data = []
    for cur_batch_id in range(len(input_metas)):
        
        path = input_metas[cur_batch_id]['img_path']

        # 提取内参矩阵的特定元素（假设是 NumPy 数组）
        cam2img = input_metas[cur_batch_id]['cam2img']
        ins = np.array([
            cam2img[0][0], cam2img[1][1], 
            cam2img[0][2], cam2img[1][2]
        ])

        cur_batch_img = img[cur_batch_id]

        cur_data = {
            'cur_batch_img': cur_batch_img, 
            'intrinsic': ins,
            'filename': path,
            'rgb': path, 
            'depth': None
        }

        test_data.append(cur_data)

    return_dict = do_scalecano_test_with_custom_data(
        model, 
        cfg,
        test_data,
        cfg.distributed,
    )
    return return_dict

        
def main_worker(local_rank: int, cfg: dict, launcher: str, test_data: list, model=None):

    logger = None
    return_dict = do_scalecano_test_with_custom_data(
        model, 
        cfg,
        test_data,
        logger,
        cfg.distributed,
        local_rank
    )
    return return_dict
    
