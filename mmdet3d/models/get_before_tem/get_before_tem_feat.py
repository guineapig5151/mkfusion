import torch
import torch.nn as nn

class get_before_tem_feat(nn.Module):
    def __init__(self, tem_num = 3):
        super().__init__()
        self.tem_num = tem_num

    def get_point_time_list(self, points):
        batch_time_point = [[] for _ in range(self.tem_num)]
        for batch_id in range(len(points)):
            cur_batch_point = points[batch_id]
            for time_id in range(1, self.tem_num):
                cur_time_mask =  (torch.abs(cur_batch_point[:, -1]) == time_id)
                cur_time_point = cur_batch_point[cur_time_mask]
                assert cur_time_point.shape[0]
                batch_time_point[time_id].append(cur_time_point)
        return batch_time_point
     
    def forward(self, batch_dict, sweep_point_list, model):
        now_batch_points = batch_dict['points'] # (batch_idx, x, y, z, i, e)
        batch_time_point = self.get_point_time_list(sweep_point_list)

        before_feat_list = []
        batch_dict_bt = {}
        for time_id in range(1, len(batch_time_point)):
            with torch.no_grad():
                batch_dict_bt['inputs'] = {}
                batch_dict_bt['inputs']['points'] = batch_time_point[time_id]
                batch_dict_bt['data_samples'] = None
                _, distill_return = model.extract_before_time_feat(batch_dict_bt)
                before_feat_list.append(distill_return)
        
        # batch_dict['points'] = batch_time_point[0]
        batch_dict['before_feat_list'] = before_feat_list
        return batch_dict