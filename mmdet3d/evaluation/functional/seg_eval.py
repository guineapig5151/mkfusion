# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
import itertools

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

def calc_conf_metrix(pred_list,
                    gt_list,
                    num_classes: int,
                    ignore_index: int = -1) -> np.ndarray:
    """
    计算混淆矩阵，输入是预测和真实标签的 list
    参数:
        pred_list: list of np.ndarray，每个 shape = (N,)
        gt_list:   list of np.ndarray，每个 shape = (N,)
        num_classes: 类别数 C
        ignore_index: 忽略的标签值（例如 -1 表示无效点）
    返回:
        conf_mat: (C, C)，混淆矩阵
    """
    assert len(pred_list) == len(gt_list), "pred_list 和 gt_list 长度必须一致"
    C = num_classes

    conf_mat = np.zeros((C, C), dtype=np.int64)
    for pred, gt in zip(pred_list, gt_list):
        assert pred.shape == gt.shape, "pred 和 gt 的形状必须一致"
        # 过滤掉 ignore_index 和 越界标签
        valid = (gt != ignore_index) & (gt >= 0) & (gt < C)
        gt_valid = gt[valid]
        pred_valid = pred[valid]

        idx = gt_valid * C + pred_valid
        conf_mat += np.bincount(idx, minlength=C*C).reshape(C, C)

    return conf_mat

def plot_confusion_matrix(cm: np.ndarray,
                          target_names: list,
                          title: str = 'Confusion matrix',
                          cmap=None,
                          normalize: bool = True,
                          save_path: str = 'confusion_matrix.png'):
    """ Visualizes a given confusion matrix.

        Arguments:
            cm: Confusion matrix.
            target_names: Class names.
            title: Plot title.
            cmap: Color map.
            normalize: Wether to normalize the data.
        """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

     #thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color='black')
                     # color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color='black')
                     # color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground truth label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return fig

def seg_eval(gt_labels, seg_preds, label2cat, ignore_index, logger=None):
    """Semantic Segmentation  Evaluation.

    Evaluate the result of the Semantic Segmentation.

    Args:
        gt_labels (list[torch.Tensor]): Ground truth labels.
        seg_preds  (list[torch.Tensor]): Predictions.
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(seg_preds) == len(gt_labels)
    num_classes = len(label2cat)

    hist_list = []
    for i in range(len(gt_labels)):
        gt_seg = gt_labels[i].astype(np.int64)
        pred_seg = seg_preds[i].astype(np.int64)

        # filter out ignored points
        pred_seg[gt_seg == ignore_index] = -1
        gt_seg[gt_seg == ignore_index] = -1

        # calculate one instance result
        hist_list.append(fast_hist(pred_seg, gt_seg, num_classes))

    # plot confusion matrix
    # plot_confusion_matrix(calc_conf_metrix(seg_preds, gt_labels, num_classes, ignore_index), target_names=[label2cat[i] for i in range(num_classes)], save_path="z_conf_mat_vod.png")

    iou = per_class_iou(sum(hist_list))
    
    # if ignore_index is in iou, replace it with nan
    if ignore_index < len(iou):
        iou[ignore_index] = np.nan
        
    miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))

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


def seg_eval_12(gt_labels, seg_preds, label2cat, ignore_index, logger=None, mask_modal=None):
    assert len(seg_preds) == len(gt_labels)
    num_classes = len(label2cat)

    hist_list = [[], []] # lidar , radar
    for i in range(len(gt_labels)):
        gt_seg = gt_labels[i].astype(np.int64)
        pred_seg = seg_preds[i].astype(np.int64)

        for modal_id in range(2):
            mask = mask_modal[modal_id]
            # filter out ignored points
            pred_seg[gt_seg == ignore_index][mask] = -1
            gt_seg[gt_seg == ignore_index][mask] = -1

            # calculate one instance result
            hist_list[modal_id].append(fast_hist(pred_seg, gt_seg, num_classes))
    for modal_id in range(2):

        iou = per_class_iou(sum(hist_list))
        
        # if ignore_index is in iou, replace it with nan
        if ignore_index < len(iou):
            iou[ignore_index] = np.nan
            
        miou = np.nanmean(iou)
        acc = get_acc(sum(hist_list))
        acc_cls = get_acc_cls(sum(hist_list))

        header = ['classes']
        for i in range(len(label2cat)):
            header.append(label2cat[i])
        header.extend(['miou', 'acc', 'acc_cls'])

        ret_dict = dict()
        table_columns = [['results']]
        table_columns.append(['modal_id: {}'.format(modal_id)])
        
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
