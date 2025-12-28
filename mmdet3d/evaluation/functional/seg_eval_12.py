# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable


def fast_hist(preds, labels, num_classes):
    """Compute the confusion matrix for every batch.

    Args:
        preds (np.ndarray):  Prediction labels of points with shape of
        (num_points, ).
        labels (np.ndarray): Ground truth labels of points with shape of
        (num_points, ).
        num_classes (int): number of classes

    Returns:
        np.ndarray: Calculated confusion matrix.
    """

    k = (labels >= 0) & (labels < num_classes)
    bin_count = np.bincount(
        num_classes * labels[k].astype(int) + preds[k],
        minlength=num_classes**2)
    return bin_count[:num_classes**2].reshape(num_classes, num_classes)


def per_class_iou(hist):
    """Compute the per class iou.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        np.ndarray: Calculated per class iou
    """

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_acc(hist):
    """Compute the overall accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated overall acc
    """

    return np.diag(hist).sum() / hist.sum()


def get_acc_cls(hist):
    """Compute the class average accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated class average acc
    """
    return np.nanmean(np.diag(hist) / hist.sum(axis=1))


def seg_eval_12(gt_labels, seg_preds, label2cat, ignore_index, logger=None, mask_lidar=None):
    assert len(seg_preds) == len(gt_labels)
    num_classes = len(label2cat)

    hist_list = [[], []] # lidar , radar
    for i in range(len(gt_labels)):
        for modal_id in range(2):
            gt_seg = gt_labels[i].astype(np.int64)
            pred_seg = seg_preds[i].astype(np.int64)           
            if modal_id == 0:
                mask = mask_lidar[i]
            else:
                mask = ~mask_lidar[i]
            pred_seg = pred_seg[mask]
            gt_seg = gt_seg[mask]

            # filter out ignored points
            pred_seg[gt_seg == ignore_index] = -1
            gt_seg[gt_seg == ignore_index] = -1

            # calculate one instance result
            hist_list[modal_id].append(fast_hist(pred_seg, gt_seg, num_classes))

    for modal_id in range(2):
        cur_modal_hist_list = hist_list[modal_id]
        iou = per_class_iou(sum(cur_modal_hist_list))
        
        # if ignore_index is in iou, replace it with nan
        if ignore_index < len(iou):
            iou[ignore_index] = np.nan
            
        miou = np.nanmean(iou)
        acc = get_acc(sum(cur_modal_hist_list))
        acc_cls = get_acc_cls(sum(cur_modal_hist_list))

        header = ['classes']
        for i in range(len(label2cat)):
            header.append(label2cat[i])
        header.extend(['miou', 'acc', 'acc_cls'])

        ret_dict = dict()
        table_columns = [['results']]
        
        for i in range(len(label2cat)):
            ret_dict[label2cat[i]] = float(iou[i])
            table_columns.append([f'{iou[i]:.4f}'])
        ret_dict['miou'] = float(miou)
        ret_dict['acc'] = float(acc)
        ret_dict['acc_cls'] = float(acc_cls)

        table_columns.append([f'{miou:.4f}'])
        table_columns.append([f'{acc:.4f}'])
        table_columns.append([f'{acc_cls:.4f}'])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

    return ret_dict
