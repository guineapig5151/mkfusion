#!/usr/bin/env python3
"""
Compute class weights (after mapping) and global RA (RCS / v_r_comp) mean/std.

Assumptions for VoD-style points:
  columns: [x, y, z, RCS, v_r, v_r_compensated, time]

Labels:
  Raw labels in seg_label_velodyne_reduced are int64 per-point; in the repo
  pipeline for VoD, labels are shifted by +1 (background: -1 -> 0). We mimic
  that by default (can be disabled via --no-add-one).

Usage example:
  python tools/stat_class_ra.py \
    --points-root data/vod/training/velodyne_reduced \
    --labels-root data/seg_label_velodyne_reduced/ \
    --mapping vod --num-classes 11 --ignore-index 11 \
    --out tools/class_ra_stats.json

The script only reads files; it does not modify any code/configs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def build_mapping(name: str) -> Tuple[np.ndarray, int]:
    """Return (mapping_array, max_label_in_raw) for a known dataset mapping.

    Currently supports 'vod' mapping used in configs/_base_/datasets/vod-3d-3class_remap.py
    mapping dict there:
        {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:11,9:9,10:10,11:8,12:11,13:11}
    """
    name = (name or '').lower()
    if name == 'vod':
        raw_max = 13
        mapping = np.zeros(raw_max + 1, dtype=np.int64)
        mapping_dict = {
            0: 0, 1: 1, 2: 2, 3: 3,
            4: 4, 5: 5, 6: 6, 7: 7,
            8: 11, 9: 9, 10: 10, 11: 8,
            12: 11, 13: 11,
        }
        for k, v in mapping_dict.items():
            mapping[k] = v
        return mapping, raw_max
    else:
        raise ValueError(f"Unknown mapping preset: {name}")


def discover_pairs(points_root: Path, labels_root: Path, ext: str) -> List[Tuple[Path, Path]]:
    """Return list of (points_file, labels_file) pairs matched by stem.
    Only files present in both roots are used.
    """
    points = {p.stem: p for p in points_root.glob(f'*{ext}')}
    labels = {p.stem: p for p in labels_root.glob(f'*{ext}')}
    common = sorted(points.keys() & labels.keys())
    return [(points[s], labels[s]) for s in common]


def main():
    ap = argparse.ArgumentParser(description='Stat class weights and RA mean/std')
    ap.add_argument('--points-root', type=str, default='data/vod/training/velodyne_reduced',
                   help='Root folder of point cloud .bin (float32 Nx7)')
    ap.add_argument('--labels-root', type=str, required=True,
                   help='Root folder of label .bin (int64 N)')
    ap.add_argument('--ext', type=str, default='.bin')
    ap.add_argument('--mapping', type=str, default='vod', help='Mapping preset name')
    ap.add_argument('--num-classes', type=int, default=11)
    ap.add_argument('--ignore-index', type=int, default=11)
    ap.add_argument('--add-one', dest='add_one', action='store_true', default=True,
                   help='Add +1 to raw labels (VoD background -1 -> 0). Default: True')
    ap.add_argument('--no-add-one', dest='add_one', action='store_false')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of files (debug)')
    ap.add_argument('--out', type=str, default='class_ra_stats.json')
    args = ap.parse_args()

    points_root = Path(args.points_root)
    labels_root = Path(args.labels_root)
    if not points_root.is_dir():
        raise FileNotFoundError(f'points_root not found: {points_root}')
    if not labels_root.is_dir():
        raise FileNotFoundError(f'labels_root not found: {labels_root}')

    mapping, raw_max = build_mapping(args.mapping)
    pairs = discover_pairs(points_root, labels_root, args.ext)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    if not pairs:
        raise RuntimeError('No matched (points, labels) files found.')

    num_classes = int(args.num_classes)
    ignore_index = int(args.ignore_index)

    # Accumulators
    cls_counts = np.zeros((num_classes,), dtype=np.int64)
    total_valid = 0
    # RA stats across valid points (exclude ignore_index)
    rcs_sum = 0.0
    rcs_sumsq = 0.0
    vr_sum = 0.0
    vr_sumsq = 0.0

    processed = 0
    for pf, lf in pairs:
        # points: float32 Nx7
        pts = np.fromfile(pf, dtype=np.float32)
        if pts.size % 7 != 0:
            # skip malformed
            continue
        pts = pts.reshape(-1, 7)
        # labels: int64 N
        lbl = np.fromfile(lf, dtype=np.int64)
        if args.add_one:
            lbl = lbl + 1  # background -1 -> 0
        if lbl.shape[0] != pts.shape[0]:
            # size mismatch, skip
            continue

        # map raw labels -> training labels
        # ensure mapping array large enough
        if mapping.shape[0] <= lbl.max():
            tmp = np.zeros((lbl.max() + 1,), dtype=np.int64)
            tmp[: mapping.shape[0]] = mapping
            tmp[mapping.shape[0]:] = ignore_index
            mapping = tmp
        m_lbl = mapping[lbl]
        valid_mask = (m_lbl != ignore_index)
        if not np.any(valid_mask):
            continue

        valid_lbl = m_lbl[valid_mask]
        # update class histogram
        hist = np.bincount(valid_lbl, minlength=num_classes)
        cls_counts += hist[:num_classes]

        # update RA global moments (RCS col=3, v_r_comp col=5)
        valid_pts = pts[valid_mask]
        rcs = valid_pts[:, 3].astype(np.float64)
        vr = valid_pts[:, 5].astype(np.float64)
        rcs_sum += rcs.sum()
        rcs_sumsq += np.square(rcs).sum()
        vr_sum += vr.sum()
        vr_sumsq += np.square(vr).sum()
        total_valid += valid_pts.shape[0]

        processed += 1

    if total_valid == 0:
        raise RuntimeError('No valid points after mapping and ignore filtering.')

    # class weights
    counts = cls_counts.astype(np.float64)
    # avoid zero division
    nonzero = counts > 0
    freq = np.zeros_like(counts)
    freq[nonzero] = counts[nonzero] / counts.sum()
    # inverse frequency (normalized to avg=1)
    inv = np.zeros_like(counts)
    inv[nonzero] = 1.0 / counts[nonzero]
    inv = inv * (counts[nonzero].mean() / inv[nonzero].mean()) if np.any(nonzero) else inv
    # median frequency balancing
    med = np.zeros_like(counts)
    if np.any(nonzero):
        med_val = np.median(counts[nonzero])
        med[nonzero] = med_val / counts[nonzero]

    # RA global mean/std
    rcs_mean = rcs_sum / total_valid
    rcs_var = max(rcs_sumsq / total_valid - rcs_mean * rcs_mean, 0.0)
    rcs_std = float(np.sqrt(rcs_var))
    vr_mean = vr_sum / total_valid
    vr_var = max(vr_sumsq / total_valid - vr_mean * vr_mean, 0.0)
    vr_std = float(np.sqrt(vr_var))

    result = {
        'samples': processed,
        'total_valid_points': int(total_valid),
        'num_classes': num_classes,
        'ignore_index': ignore_index,
        'class_counts': cls_counts.tolist(),
        'class_freq': freq.tolist(),
        'class_weight_inverse': inv.tolist(),
        'class_weight_median_freq': med.tolist(),
        'ra_global_moments': {
            'rcs_mean': float(rcs_mean), 'rcs_std': rcs_std,
            'vr_mean': float(vr_mean),   'vr_std':  vr_std,
        }
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

