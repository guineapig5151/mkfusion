auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
class_names = [
    'background',
    'Car',
    'Pedestrian',
    'Cyclist',
    'bicycle',
    'bicycle_rack',
    'moped_scooter',
    'rider',
    'motor',
    'truck',
    'ride_other',
]
data_root = 'data/vod/lidar_in_radarcoord/'
dataset_type = 'VodDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
labels_map = dict({
    0: 0,
    1: 1,
    10: 10,
    11: 8,
    12: 11,
    13: 11,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 11,
    9: 9
})
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.008
metainfo = dict(
    classes=[
        'background',
        'Car',
        'Pedestrian',
        'Cyclist',
        'bicycle',
        'bicycle_rack',
        'moped_scooter',
        'rider',
        'motor',
        'truck',
        'ride_other',
    ],
    max_label=13,
    seg_label_mapping=dict({
        0: 0,
        1: 1,
        10: 10,
        11: 8,
        12: 11,
        13: 11,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 11,
        9: 9
    }))
model = dict(
    backbone=dict(
        base_channels=32,
        block_type='basic',
        decoder_blocks=[
            2,
            2,
            2,
            2,
        ],
        decoder_channels=[
            256,
            128,
            96,
            96,
        ],
        encoder_blocks=[
            2,
            2,
            2,
            2,
        ],
        encoder_channels=[
            32,
            64,
            128,
            256,
        ],
        in_channels=7,
        num_stages=4,
        sparseconv_backend='spconv',
        type='MinkUNetBackbone'),
    data_preprocessor=dict(
        batch_first=True,
        max_voxels=80000,
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=-1,
            max_voxels=(
                -1,
                -1,
            ),
            point_cloud_range=[
                0,
                -25.6,
                -3,
                51.2,
                25.6,
                2,
            ],
            voxel_size=[
                0.05,
                0.05,
                0.125,
            ]),
        voxel_type='minkunet_multi'),
    decode_head=dict(
        channels=96,
        dropout_ratio=0,
        ignore_index=11,
        loss_decode=dict(avg_non_ignore=True, type='mmdet.CrossEntropyLoss'),
        num_classes=11,
        type='MinkUNetHead'),
    test_cfg=dict(),
    train_cfg=dict(),
    type='MinkUNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.008, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            24,
            32,
        ],
        type='MultiStepLR'),
]
point_cloud_range = [
    0,
    -25.6,
    -3,
    51.2,
    25.6,
    2,
]
pts_semantic_label_path = 'data/vod_seg_label_all/'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/lidar_radar'),
        data_root='data/vod/lidar_in_radarcoord/',
        filter_empty_gt=False,
        metainfo=dict(
            classes=[
                'background',
                'Car',
                'Pedestrian',
                'Cyclist',
                'bicycle',
                'bicycle_rack',
                'moped_scooter',
                'rider',
                'motor',
                'truck',
                'ride_other',
            ],
            max_label=13,
            seg_label_mapping=dict({
                0: 0,
                1: 1,
                10: 10,
                11: 8,
                12: 11,
                13: 11,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 11,
                9: 9
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=7,
                type='LoadPointsFromFile',
                use_dim=7),
            dict(
                dataset_type='vod',
                pts_semantic_label_path=
                'data/vod_seg_label_all/',
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                point_cloud_range=[
                    0,
                    -25.6,
                    -3,
                    51.2,
                    25.6,
                    2,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='VodDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='SegMetric', data_type='12')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(
        dataset_type='vod',
        pts_semantic_label_path=
        'data/vod_seg_label_all/',
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        point_cloud_range=[
            0,
            -25.6,
            -3,
            51.2,
            25.6,
            2,
        ],
        type='PointsRangeFilter'),
    dict(keys=[
        'points',
        'pts_semantic_mask',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/lidar_radar'),
            data_root='data/vod/lidar_in_radarcoord/',
            filter_empty_gt=False,
            metainfo=dict(
                classes=[
                    'background',
                    'Car',
                    'Pedestrian',
                    'Cyclist',
                    'bicycle',
                    'bicycle_rack',
                    'moped_scooter',
                    'rider',
                    'motor',
                    'truck',
                    'ride_other',
                ],
                max_label=13,
                seg_label_mapping=dict({
                    0: 0,
                    1: 1,
                    10: 10,
                    11: 8,
                    12: 11,
                    13: 11,
                    2: 2,
                    3: 3,
                    4: 4,
                    5: 5,
                    6: 6,
                    7: 7,
                    8: 11,
                    9: 9
                })),
            modality=dict(use_camera=False, use_lidar=True),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=7,
                    type='LoadPointsFromFile',
                    use_dim=7),
                dict(
                    dataset_type='vod',
                    pts_semantic_label_path=
                    'data/vod_seg_label_all/',
                    type='LoadAnnotations3D',
                    with_bbox_3d=False,
                    with_label_3d=False,
                    with_seg_3d=True),
                dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
                dict(
                    rot_range=[
                        -0.78539816,
                        0.78539816,
                    ],
                    scale_ratio_range=[
                        0.95,
                        1.05,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(type='PointSegClassMapping'),
                dict(
                    point_cloud_range=[
                        0,
                        -25.6,
                        -3,
                        51.2,
                        25.6,
                        2,
                    ],
                    type='PointsRangeFilter'),
                dict(
                    keys=[
                        'points',
                        'pts_semantic_mask',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='VodDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(
        dataset_type='vod',
        pts_semantic_label_path=
        'data/vod_seg_label_all/',
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        type='GlobalRotScaleTrans'),
    dict(type='PointSegClassMapping'),
    dict(
        point_cloud_range=[
            0,
            -25.6,
            -3,
            51.2,
            25.6,
            2,
        ],
        type='PointsRangeFilter'),
    dict(keys=[
        'points',
        'pts_semantic_mask',
    ], type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/lidar_radar'),
        data_root='data/vod/lidar_in_radarcoord/',
        filter_empty_gt=False,
        metainfo=dict(
            classes=[
                'background',
                'Car',
                'Pedestrian',
                'Cyclist',
                'bicycle',
                'bicycle_rack',
                'moped_scooter',
                'rider',
                'motor',
                'truck',
                'ride_other',
            ],
            max_label=13,
            seg_label_mapping=dict({
                0: 0,
                1: 1,
                10: 10,
                11: 8,
                12: 11,
                13: 11,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 11,
                9: 9
            })),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=7,
                type='LoadPointsFromFile',
                use_dim=7),
            dict(
                dataset_type='vod',
                pts_semantic_label_path=
                'data/vod_seg_label_all/',
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True),
            dict(type='PointSegClassMapping'),
            dict(
                point_cloud_range=[
                    0,
                    -25.6,
                    -3,
                    51.2,
                    25.6,
                    2,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'pts_semantic_mask',
                ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='VodDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(data_type='12', type='SegMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
