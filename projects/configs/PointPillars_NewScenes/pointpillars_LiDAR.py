# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
point_cloud_range = [-60, -40, -3.0, 60, 40, 5.0] 
voxel_size = [0.25, 0.25, 8]

# For newScenes we usually do 4-class detection
class_names = [
    'car', 'pedestrian', 'rider', 'large_vehicle']

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False)

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

#----model------------

model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=64, 
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4, 
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 480]), 
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=4, 
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-60, -40, 0.9104247242165809, 60, 40, 0.9104247242165809],
                [-60, -40, 1.1421614665993767, 60, 40, 1.1421614665993767],
                [-60, -40, 0.9059764319390522, 60, 40, 0.9059764319390522],
                [-60, -40, 1.5158325603046292, 60, 40, 1.5158325603046292]
            ],
            sizes=[
                [1.9768212501227105,4.637021209998035,1.6647611354273741],  # car
                [0.796163784946599, 0.8183815295280997,1.6895737765415433],  # pedestrian
                [0.912318683145357, 1.9201067650572057,1.620921669034068],  # rider
                [2.6724696700336494, 8.184714524976142, 3.0254503871391982],  # large_vehicle
                # [2.4560939, 10.73778078, 2.73004906]
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))

#-------dataset----------
dataset_type = 'NewScenesDataset' #---dataset
data_root = 'data/NewScenes_Final/' 
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
    type='LoadPointsFromFile',
    coord_type='LIDAR',
    load_dim=6,
    use_dim=4,
    file_client_args=file_client_args),

    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d','points']) 
]

test_pipeline = [
    
    dict(
    type='LoadPointsFromFile',
    coord_type='LIDAR',
    load_dim=6,
    use_dim=4,
    file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080), 
        pts_scale_ratio=1,
        flip=False,
        transforms=[
    
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['points']) 
        ])
]

data = dict(
    samples_per_gpu=2,  
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'newscenes-final_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality, 
        test_mode=False,
        use_valid_flag=True,  
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'newscenes-final_infos_temporal_val.pkl',
             pipeline=test_pipeline,  
             classes=class_names, 
             modality=input_modality, 
             samples_per_gpu=1),
    test=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'newscenes-final_infos_temporal_val.pkl',
             pipeline=test_pipeline,  
             classes=class_names, 
             modality=input_modality, 
             samples_per_gpu=1),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

evaluation = dict(interval=1, pipeline=test_pipeline) 
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[20, 23])
momentum_config = None
log_config = dict(
    interval=50, #---50
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


total_epochs = 24
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/NewScenes_Final/pointpillars_LiDAR'

load_from = None
resume_from = None
workflow = [('train', 1)]
use_old_custom_multi_gpu_test=True
