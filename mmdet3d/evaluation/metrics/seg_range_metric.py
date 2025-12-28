# Copyright (c) OpenMMLab. All rights reserved.
"""
Range-binned 3D semantic segmentation metric for failure analysis.

This metric computes per-bin confusion matrices and mIoU over distance
intervals (e.g., <20m, 20-40m, >40m). It mirrors the behavior of
SegMetric but augments it with distance-aware breakdowns.

Typical usage in a config:

    val_evaluator = dict(
        type='SegRangeMetric',
        dist_bins=[0, 20, 40, float('inf')],
        dist_mode='xy',  # use sqrt(x^2 + y^2) by default
        per_class=True,
    )

    Notes:
    - We rely on data_batch['inputs']['points'][i] to compute per-point distance.
    - Points are expected to be in the same order as gt/pred labels.
    - Dynamic/static metrics: we assume the 6th feature (0-based index 5) is
      per-point velocity magnitude; points with |v| > dyn_speed_thr are treated
      as dynamic.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet3d.registry import METRICS
from mmdet3d.evaluation.functional.seg_eval import (
    fast_hist, per_class_iou, get_acc, get_acc_cls,
)


def _as_numpy(x) -> np.ndarray:
    """Best-effort convert tensor/list to numpy array without copy when possible."""
    # torch is optional at runtime here
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return np.array(x) if not isinstance(x, np.ndarray) else x


def _compute_range(points_xyz: np.ndarray, mode: str = 'xy') -> np.ndarray:
    """Compute per-point distance in the chosen mode.

    Args:
        points_xyz: (N, >=3) array, columns 0..2 are x,y,z in LiDAR/ego frame.
        mode: 'xy' for planar range sqrt(x^2+y^2); 'xyz' for euclidean 3D.
    Returns:
        (N,) distances in meters.
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] >= 3, \
        'points must be (N, C) with at least 3 columns (x,y,z).'
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    if mode == 'xyz':
        z = points_xyz[:, 2]
        return np.sqrt(x * x + y * y + z * z)
    # default: 'xy'
    return np.sqrt(x * x + y * y)


@METRICS.register_module()
class SegRangeMetric(BaseMetric):
    """Distance-binned 3D semantic segmentation evaluation metric.

    Args:
        dist_bins (Sequence[float]): Monotonically increasing bin edges.
            Example: [0, 20, 40, float('inf')]. Default: [0, 20, 40, inf].
        dist_mode (str): One of {'xy', 'xyz'}. Default: 'xy'.
        per_class (bool): Whether to report per-class IoU per bin. Default: True.
        collect_device (str): 'cpu' or 'gpu' for distributed collection.
        prefix (str | None): Metric name prefix. Default: None.
    """

    default_prefix: Optional[str] = 'seg_range'

    def __init__(
        self,
        dist_bins: Sequence[float] = (0.0, 20.0, 40.0, float('inf')),
        dist_mode: str = 'xy',
        per_class: bool = True,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.dist_bins = np.asarray(dist_bins, dtype=float)
        assert np.all(np.diff(self.dist_bins) > 0), 'dist_bins must be strictly increasing.'
        assert dist_mode in ('xy', 'xyz'), "dist_mode must be 'xy' or 'xyz'"
        self.dist_mode = dist_mode
        self.per_class = per_class
        # dynamic/static config (0-based 5 means the 6th column)
        self.dyn_vel_index: int = kwargs.pop('dyn_vel_index', 5)
        self.dyn_speed_thr: float = kwargs.pop('dyn_speed_thr', 0.1)
        self.dyn_abs: bool = kwargs.pop('dyn_abs', True)

    # Each item we push into self.results is a tuple: (gt, pred, dist)
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for bidx, data_sample in enumerate(data_samples):
            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']

            # gt / pred as numpy int64
            gt = _as_numpy(eval_ann_info['pts_semantic_mask']).astype(np.int64)
            pred = _as_numpy(pred_3d['pts_semantic_mask']).astype(np.int64)

            # points come from data_batch. Keep only xyz columns.
            pts = data_batch['inputs']['points'][bidx]
            pts = _as_numpy(pts)
            if pts.shape[0] != gt.shape[0]:
                raise ValueError(
                    f'points count ({pts.shape[0]}) != labels count ({gt.shape[0]}). '
                    'Ensure Pack3DDetInputs keeps the same ordering.')
            d = _compute_range(pts[:, :3], self.dist_mode)
            # dynamic mask based on velocity feature
            try:
                vel = pts[:, self.dyn_vel_index]
                if self.dyn_abs:
                    dyn_mask = np.abs(vel) > float(self.dyn_speed_thr)
                else:
                    dyn_mask = vel > float(self.dyn_speed_thr)
                dyn_mask = dyn_mask.astype(bool)
            except Exception:
                # Fallback: no dynamic info
                dyn_mask = None
            self.results.append((gt, pred, d, dyn_mask))

    def _format_bin_name(self, lo: float, hi: float) -> str:
        def _f(x: float) -> str:
            if np.isinf(x):
                return 'Inf'
            if abs(x - int(x)) < 1e-6:
                return str(int(x))
            return f'{x:g}'
        return f'{_f(lo)}-{_f(hi)}m'

    def compute_metrics(self, results: List[tuple]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']
        num_classes = len(label2cat)

        # Prepare per-bin accumulators
        B = len(self.dist_bins) - 1
        hist_bins = [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in range(B)]
        count_bins = np.zeros(B, dtype=np.int64)

        # Accumulate per-bin confusion matrices
        # Also accumulate dynamic/static confusion matrices (global, not per-bin)
        hist_dyn = np.zeros((num_classes, num_classes), dtype=np.int64)
        hist_sta = np.zeros((num_classes, num_classes), dtype=np.int64)
        count_dyn_all = 0
        count_sta_all = 0
        # Per-range dynamic/static point counts (raw counts, not filtering ignore)
        count_dyn_bins = np.zeros(B, dtype=np.int64)
        count_sta_bins = np.zeros(B, dtype=np.int64)
        dyn_present = False

        for item in results:
            # backward-compat unpacking
            if len(item) == 4:
                gt, pred, dist, dyn_mask = item
            else:
                gt, pred, dist = item
                dyn_mask = None
            # Map ignored GT to -1 in pred and gt for fair counting
            gt = gt.astype(np.int64)
            pred = pred.astype(np.int64)
            pred[gt == ignore_index] = -1
            gt[gt == ignore_index] = -1

            # Bin assignment. np.digitize returns indices in 1..B; shift to 0..B-1
            bin_ids = np.digitize(dist, self.dist_bins, right=False) - 1
            # Clamp to valid range just in case of numerical edge cases
            bin_ids = np.clip(bin_ids, 0, B - 1)

            for b in range(B):
                mask = (bin_ids == b)
                if not np.any(mask):
                    continue
                count_bins[b] += int(mask.sum())
                # Extract valid labels within the mask
                gtb = gt[mask]
                predb = pred[mask]
                hist_bins[b] += fast_hist(predb, gtb, num_classes)

            # Global dynamic/static accumulators
            if dyn_mask is not None:
                # ensure shape matches
                if dyn_mask.shape[0] == gt.shape[0]:
                    dyn_present = True
                    dmask = dyn_mask
                    smask = ~dyn_mask
                    count_dyn_all += int(dmask.sum())
                    count_sta_all += int(smask.sum())
                    # compute confusion matrices under the two masks
                    hist_dyn += fast_hist(pred[dmask], gt[dmask], num_classes)
                    hist_sta += fast_hist(pred[smask], gt[smask], num_classes)
                    # per-range dynamic/static counts
                    for b in range(B):
                        mb = (bin_ids == b)
                        if not np.any(mb):
                            continue
                        count_dyn_bins[b] += int((mb & dmask).sum())
                        count_sta_bins[b] += int((mb & smask).sum())

        # Build tables and return dict
        ret: Dict[str, float] = {}
        headers = ['classes'] + [label2cat[i] for i in range(num_classes)] + ['miou', 'acc', 'acc_cls', 'num_pts', 'ratio']

        table_rows = []
        for b in range(B):
            lo, hi = self.dist_bins[b], self.dist_bins[b + 1]
            bin_name = self._format_bin_name(lo, hi)
            hist = hist_bins[b]

            iou = per_class_iou(hist)
            if ignore_index < len(iou):
                iou[ignore_index] = np.nan
            miou = float(np.nanmean(iou))
            acc = float(get_acc(hist)) if hist.sum() > 0 else 0.0
            acc_cls = float(get_acc_cls(hist)) if hist.sum() > 0 else 0.0

            # Fill return dict
            ret[f'miou@{bin_name}'] = miou
            ret[f'acc@{bin_name}'] = acc
            ret[f'acc_cls@{bin_name}'] = acc_cls
            ret[f'num_pts@{bin_name}'] = int(count_bins[b])
            total_pts = int(count_bins.sum())
            ratio = float(count_bins[b]) / total_pts if total_pts > 0 else 0.0
            ret[f'ratio@{bin_name}'] = ratio

            # Prepare table row
            row = [bin_name]
            for c in range(num_classes):
                val = float(iou[c]) if not np.isnan(iou[c]) else float('nan')
                if self.per_class:
                    # also export per-class iou to ret
                    ret[f'{label2cat[c]}@{bin_name}'] = val
                row.append(f'{val:.4f}' if not np.isnan(val) else 'nan')
            row += [f'{miou:.4f}', f'{acc:.4f}', f'{acc_cls:.4f}', str(int(count_bins[b])) , f'{ratio:.4f}']
            table_rows.append(row)

        # Aggregate '@all' across all bins to match SegMetric overall behavior
        hist_all = np.zeros((num_classes, num_classes), dtype=np.int64)
        for h in hist_bins:
            hist_all += h
        iou_all = per_class_iou(hist_all)
        if ignore_index < len(iou_all):
            iou_all[ignore_index] = np.nan
        if hist_all.sum() > 0:
            miou_all = float(np.nanmean(iou_all))
            acc_all = float(get_acc(hist_all))
            acc_cls_all = float(get_acc_cls(hist_all))
        else:
            miou_all = 0.0
            acc_all = 0.0
            acc_cls_all = 0.0
        ret['miou@all'] = miou_all
        ret['acc@all'] = acc_all
        ret['acc_cls@all'] = acc_cls_all
        total_pts = int(count_bins.sum())
        ret['num_pts@all'] = total_pts
        ret['ratio@all'] = 1.0 if total_pts > 0 else 0.0
        # Per-class @all
        row_all = ['all']
        for c in range(num_classes):
            val = float(iou_all[c]) if not np.isnan(iou_all[c]) else float('nan')
            if self.per_class:
                ret[f'{label2cat[c]}@all'] = val
            row_all.append(f'{val:.4f}' if not np.isnan(val) else 'nan')
        row_all += [f'{miou_all:.4f}', f'{acc_all:.4f}', f'{acc_cls_all:.4f}', str(total_pts), f'{(1.0 if total_pts > 0 else 0.0):.4f}']
        table_rows.append(row_all)

        # Pretty print once
        table_data = [headers] + table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        # One-line highlighted ratios for quick glance
        total_pts_line = int(count_bins.sum())
        parts = []
        for b in range(B):
            lo, hi = self.dist_bins[b], self.dist_bins[b + 1]
            bn = self._format_bin_name(lo, hi)
            ptsb = int(count_bins[b])
            rb = float(ptsb) / total_pts_line if total_pts_line > 0 else 0.0
            parts.append(f'{bn}:{rb:.4f}({ptsb})')
        ratio_all_line = 1.0 if total_pts_line > 0 else 0.0
        print_log('[SegRange] point ratios | ' + ' | '.join(parts) +
                  f' | all:{ratio_all_line:.4f}({total_pts_line})',
                  logger=logger)

        # Per-range dynamic/static point counts (highlighted; independent of IoU)
        if dyn_present:
            parts_ds = []
            for b in range(B):
                lo, hi = self.dist_bins[b], self.dist_bins[b + 1]
                bn = self._format_bin_name(lo, hi)
                parts_ds.append(f'{bn} dyn:{int(count_dyn_bins[b])} sta:{int(count_sta_bins[b])}')
            print_log('[SegRange] dyn/static per-range | ' + ' | '.join(parts_ds) +
                      f' | all dyn:{int(count_dyn_all)} sta:{int(count_sta_all)}',
                      logger=logger)

        # Dynamic / Static overall metrics (do not affect range metrics)
        if (hist_dyn.sum() + hist_sta.sum()) > 0:
            # dynamic
            iou_dyn = per_class_iou(hist_dyn)
            if ignore_index < len(iou_dyn):
                iou_dyn[ignore_index] = np.nan
            miou_dyn = float(np.nanmean(iou_dyn)) if hist_dyn.sum() > 0 else 0.0
            acc_dyn = float(get_acc(hist_dyn)) if hist_dyn.sum() > 0 else 0.0
            acc_cls_dyn = float(get_acc_cls(hist_dyn)) if hist_dyn.sum() > 0 else 0.0
            ret['miou@dynamic'] = miou_dyn
            ret['acc@dynamic'] = acc_dyn
            ret['acc_cls@dynamic'] = acc_cls_dyn
            ret['num_pts@dynamic'] = int(count_dyn_all)

            # static
            iou_sta = per_class_iou(hist_sta)
            if ignore_index < len(iou_sta):
                iou_sta[ignore_index] = np.nan
            miou_sta = float(np.nanmean(iou_sta)) if hist_sta.sum() > 0 else 0.0
            acc_sta = float(get_acc(hist_sta)) if hist_sta.sum() > 0 else 0.0
            acc_cls_sta = float(get_acc_cls(hist_sta)) if hist_sta.sum() > 0 else 0.0
            ret['miou@static'] = miou_sta
            ret['acc@static'] = acc_sta
            ret['acc_cls@static'] = acc_cls_sta
            ret['num_pts@static'] = int(count_sta_all)

            total_ds = count_dyn_all + count_sta_all
            r_dyn = (float(count_dyn_all) / total_ds) if total_ds > 0 else 0.0
            r_sta = (float(count_sta_all) / total_ds) if total_ds > 0 else 0.0
            print_log('[SegRange] dyn/static | '
                      f'mIoU_dyn:{miou_dyn:.4f} acc_dyn:{acc_dyn:.4f} '
                      f'| mIoU_sta:{miou_sta:.4f} acc_sta:{acc_sta:.4f} '
                      f'| pts dyn:{count_dyn_all}({r_dyn:.4f}) sta:{count_sta_all}({r_sta:.4f})',
                      logger=logger)

        return ret
