import torch

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data
con = 'mmdet3d/models/segmentors/mono/configs/HourglassDecoder/vit.raft5.giant2.py'
load = 'metric3d_v2/weight/metric_depth_vit_giant2_800k.pth'

cfg = Config.fromfile(con)
cfg.load_from = load
cfg.show_dir = 'mmdet3d/models/segmentors/mono/'

# load data info
data_info = {}
load_data_info('data_info', data_info=data_info)
cfg.mldb_info = data_info

model = get_configured_monodepth_model(cfg)
model = torch.nn.DataParallel(model).cuda()
model, _,  _, _ = load_ckpt(load, model, strict_match=False)
model.eval()

test_data = [{'rgb': 'data/vod/training/image_2/04053.jpg', 'depth': None, 'intrinsic': [1495.468642, 1495.468642, 961.272442, 624.89592], 'filename': '04053.jpg', 'folder': 'data'}]

do_scalecano_test_with_custom_data(
    model, 
    cfg,
    test_data,
    logger=None,
    is_distributed=False,
    local_rank=0,
)