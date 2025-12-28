# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmengine import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast, load_checkpoint

from mmdet3d.registry import MODELS
from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', type=int, default=2000,
                        help='samples (iterations) to benchmark')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--warmup', type=int, default=5,
        help='number of warmup iterations skipped for timing')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Whether to use automatic mixed precision inference')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--e2e', action='store_true',
        help='Measure end-to-end latency (include dataloader & CPU transforms)')
    parser.add_argument(
        '--percentiles', type=str, default='50,90,95,99',
        help='Comma-separated latency percentiles to report, e.g. 50,90,99')
    parser.add_argument(
        '--mem', action='store_true',
        help='Report peak CUDA memory allocated/reserved (MiB)')
    parser.add_argument(
        '--save-csv', type=str, default=None,
        help='Optional path to save per-iteration latency (csv)')
    parser.add_argument(
        '--params-used-only', action='store_true',
        help='Report only parameters used at runtime (via forward hooks)')
    parser.add_argument(
        '--param-scan-iters', type=int, default=1,
        help='How many measured iterations to scan for used params before removing hooks')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_default_scope('mmdet3d')

    # build config and set cudnn_benchmark
    cfg = Config.fromfile(args.config)

    if cfg.env_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # force test batch size = 1 (ignore config)
    try:
        cfg.test_dataloader['batch_size'] = 1
        print('[benchmark] Forcing cfg.test_dataloader.batch_size = 1')
    except Exception as e:
        print(f'[benchmark] Warning: cannot force batch_size=1: {e}')

    # build dataloader
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # build model and load checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    model.to(get_device())
    model.eval()

    # ---- Parameter statistics (after optional conv-bn fusion) ----
    # If user requests runtime-used-only stats, defer printing until after hooks collect usage
    if not args.params_used_only:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        frozen_params = total_params - trainable_params
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(
            'Parameter stats -> '
            f'total: {total_params/1e6:.3f}M | '
            f'trainable: {trainable_params/1e6:.3f}M | '
            f'frozen: {frozen_params/1e6:.3f}M | '
            f'size: {param_bytes/1024**2:.2f} MiB')
    else:
        print('Parameter stats: runtime-used-only mode enabled; collecting during warmup/prescan...')

    # We'll reset CUDA peak memory right before measured iterations begin

    # the first several iterations may be very slow so skip them
    num_warmup = max(int(args.warmup), 0)
    pure_inf_time = 0.0  # forward-only time
    e2e_time = 0.0       # optional end-to-end time (include dataloader)
    per_iter_forward_ms = []
    per_iter_e2e_ms = []
    # runtime-used params tracking (collect via forward hooks on modules that own params)
    used_param_ids = set()
    hook_handles = []
    collecting_used_params = args.params_used_only or False
    remaining_scan = 0
    scan_in_warmup = collecting_used_params and num_warmup > 0
    if collecting_used_params:
        remaining_scan = int(args.param_scan_iters)
        if scan_in_warmup:
            remaining_scan = min(remaining_scan, num_warmup)

    def mark_used_params(mod, _inputs):
        # only record for modules that own parameters directly
        if not collecting_used_params:
            return
        for p in mod.parameters(recurse=False):
            used_param_ids.add(id(p))
    # infer effective batch size for latency per sample
    bs = getattr(dataloader, 'batch_size', None)
    if bs is None:
        try:
            bs = cfg.test_dataloader.get('batch_size', 1)
        except Exception:
            bs = 1

    # helper for device-aware sync
    def device_sync():
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    # attach hooks only if we plan to collect used params
    if collecting_used_params:
        for m in model.modules():
            has_local_params = False
            for _ in m.parameters(recurse=False):
                has_local_params = True
                break
            if has_local_params:
                try:
                    h = m.register_forward_pre_hook(mark_used_params)
                    hook_handles.append(h)
                except Exception:
                    pass

    # If we have no warmup but need to scan used params, run a short pre-scan
    # that is excluded from measured iterations to avoid affecting FPS/latency.
    if collecting_used_params and not scan_in_warmup and remaining_scan > 0:
        pre_iter = iter(dataloader)
        for _ in range(remaining_scan):
            try:
                data = next(pre_iter)
            except StopIteration:
                break
            # run a prescan forward without timing accumulation
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                with torch.inference_mode():
                    with autocast(enabled=args.amp):
                        model.test_step(data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        # remove hooks after prescan
        collecting_used_params = False
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        hook_handles.clear()
        # reset CUDA peak memory just before measured window
        if args.mem and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    # benchmark with several samples and take the average
    data_iter = iter(dataloader)
    for i in range(args.samples):
        # End-to-end timing starts before fetching the batch if enabled
        e2e_start = time.perf_counter() if args.e2e else None

        try:
            data = next(data_iter)
        except StopIteration:
            # dataset shorter than requested samples
            if i == 0:
                print('No data from dataloader. Check test_dataloader config/path.')
            break

        device_sync()
        fwd_start = time.perf_counter()

        # Use no-grad & inference mode to better match production inference
        with torch.inference_mode():
            with autocast(enabled=args.amp):
                model.test_step(data)

        device_sync()
        fwd_elapsed = time.perf_counter() - fwd_start
        e2e_elapsed = None
        if args.e2e and e2e_start is not None:
            e2e_elapsed = time.perf_counter() - e2e_start

        # If scanning during warmup, stop and clean hooks without affecting measured window
        if scan_in_warmup and i < num_warmup and remaining_scan > 0:
            remaining_scan -= 1
            if remaining_scan == 0:
                for h in hook_handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
                hook_handles.clear()
                scan_in_warmup = False
                collecting_used_params = False
                # reset CUDA peak memory just before measured window
                if args.mem and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

        if i >= num_warmup:
            pure_inf_time += fwd_elapsed
            per_iter_forward_ms.append(fwd_elapsed * 1000.0)
            if e2e_elapsed is not None:
                e2e_time += e2e_elapsed
                per_iter_e2e_ms.append(e2e_elapsed * 1000.0)

            if (i + 1) % args.log_interval == 0:
                processed = (i + 1 - num_warmup)
                fps = processed / pure_inf_time
                avg_ms_batch = 1000.0 * pure_inf_time / processed
                avg_ms_sample = avg_ms_batch / (bs if bs else 1)
                msg = (
                    f'Done iter [{i + 1:<3}/ {args.samples}], '
                    f'fps: {fps:.1f} iter/s | '
                    f'latency: {avg_ms_batch:.2f} ms/batch, '
                    f'{avg_ms_sample:.2f} ms/sample @bs={bs}'
                )
                if args.e2e and per_iter_e2e_ms:
                    avg_ms_e2e = 1000.0 * e2e_time / processed
                    msg += f' | e2e: {avg_ms_e2e:.2f} ms/batch'
                print(msg)

    # summary
    processed = max(0, len(per_iter_forward_ms))
    if processed > 0:
        fps = processed / pure_inf_time
        avg_ms_batch = 1000.0 * pure_inf_time / processed
        avg_ms_sample = avg_ms_batch / (bs if bs else 1)
        msg = (
            f'Overall -> fps: {fps:.1f} iter/s | '
            f'latency: {avg_ms_batch:.2f} ms/batch, '
            f'{avg_ms_sample:.2f} ms/sample @bs={bs}'
        )
        if args.e2e and per_iter_e2e_ms:
            avg_ms_e2e = 1000.0 * e2e_time / processed
            msg += f' | e2e: {avg_ms_e2e:.2f} ms/batch'
        print(msg)

        # Runtime-used parameter stats (if requested or if hooks were attached)
        if used_param_ids:
            # compute only over used params
            used_total = 0
            used_trainable = 0
            used_bytes = 0
            for p in model.parameters():
                if id(p) in used_param_ids:
                    n = p.numel()
                    used_total += n
                    used_bytes += n * p.element_size()
                    if p.requires_grad:
                        used_trainable += n
            used_frozen = used_total - used_trainable
            print(
                'Runtime-used params -> '
                f'total: {used_total/1e6:.3f}M | '
                f'trainable: {used_trainable/1e6:.3f}M | '
                f'frozen: {used_frozen/1e6:.3f}M | '
                f'size: {used_bytes/1024**2:.2f} MiB')

        # percentiles
        try:
            import numpy as np
            ps = [int(p.strip()) for p in str(args.percentiles).split(',') if p.strip()]
            fwd_vals = np.array(per_iter_forward_ms)
            pvals = np.percentile(fwd_vals, ps)
            pmsg = 'Forward percentiles (ms): ' + ', '.join(
                f'P{p}={v:.2f}' for p, v in zip(ps, pvals))
            print(pmsg)
            if per_iter_e2e_ms:
                e2e_vals = np.array(per_iter_e2e_ms)
                epvals = np.percentile(e2e_vals, ps)
                epmsg = 'E2E percentiles (ms): ' + ', '.join(
                    f'P{p}={v:.2f}' for p, v in zip(ps, epvals))
                print(epmsg)
        except Exception:
            pass

        # memory report
        if args.mem and torch.cuda.is_available():
            try:
                peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
                print(f'CUDA peak memory -> allocated: {peak_alloc:.2f} MiB | '
                      f'reserved: {peak_reserved:.2f} MiB')
            except Exception:
                pass

        # optional CSV save
        if args.save_csv:
            try:
                import csv
                with open(args.save_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['iter', 'forward_ms']
                    if per_iter_e2e_ms:
                        header.append('e2e_ms')
                    writer.writerow(header)
                    for idx, fwd_ms in enumerate(per_iter_forward_ms, start=1):
                        row = [idx, fwd_ms]
                        if per_iter_e2e_ms:
                            row.append(per_iter_e2e_ms[idx-1])
                        writer.writerow(row)
                print(f'Saved per-iteration latency CSV to {args.save_csv}')
            except Exception as e:
                print(f'Failed to save CSV: {e}')
    else:
        print('Overall -> not enough samples after warmup to compute latency/FPS. '
              'Increase --samples or reduce --warmup.')


if __name__ == '__main__':
    main()
