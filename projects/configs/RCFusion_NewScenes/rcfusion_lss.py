# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------
# smaller input size: 1920*1080 -> (1920*1080)*0.5

point_cloud_range = [-60.0, -40.0, -3.0, 60.0, 40.0, 5.0] 
voxel_size = [0.25, 0.25, 8]

## For newScenes we usually do 9-class detection
class_names = [
    'car', 'pedestrian', 'rider', 'large_vehicle']
# Input modality for newScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True) 


plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) 


#----model------------
final_dim=(544, 960) 
downsample=4
model = dict(
    type='RCFusion_FasterRCNN',
    freeze_img=False,
    se=True,
    rc_fusion='cross_attention',
    camera_stream=True, 
    lss=False,  
    grid=0.5, 
    num_views=6,
    final_dim=final_dim, 
    pc_range=point_cloud_range, 
    downsample=downsample,  #--8
    camera_depth_range=[1,60,1],
    img_depth_loss_method='kld',
    img_depth_loss_weight=1.0,
    
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNet',
        in_channels=7, 
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        with_velocity_snr_center=True,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
        ),
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


    img_backbone=dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(1,2,3),
    frozen_stages=1,
    norm_cfg=dict(type='SyncBN', requires_grad=False), 
    norm_eval=True, 
    style='pytorch'),

    img_neck=dict(
        type='FPNC',
        final_dim=final_dim,
        downsample=downsample, 
        in_channels=[512,1024,2048],
        out_channels=256,
        use_adp=True,
        num_outs=4),

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
                [2.6724696700336494, 8.184714524976142, 3.0254503871391982]  # large_vehicle
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
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2], #---
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
data_root = 'data/NewScenes_Final/' #---数据路径
file_client_args = dict(backend='disk')

radar_use_dims = [0, 1, 2, 3, 4, 5, 6, 7]
train_pipeline = [
    
    dict(
    type='LoadRadarPointsMultiSweeps',
    load_dim=8,
    sweeps_num=3,
    use_dim=radar_use_dims,
    file_client_args=file_client_args,
    max_num=40000, #--没用到_pad_or_drop
    pc_range=point_cloud_range),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadMultiViewImageFromFiles_newsc', to_float32=True),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),

    dict(type='NormalizeMultiviewImage', **img_norm_cfg), #----normalization-------
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
        #----加入gt_depth-------
    dict(type='LoadGTDepth',scale=0.5),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d','img','points','img_depth']) #---加入radar points---
]

test_pipeline = [
    
    dict(
    type='LoadRadarPointsMultiSweeps',
    load_dim=8,
    sweeps_num=3,
    use_dim=radar_use_dims,
    file_client_args=file_client_args,
    max_num=40000,
    pc_range=point_cloud_range),

    dict(type='LoadMultiViewImageFromFiles_newsc', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080), 
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img','points']) 
        ])
]


#------------val和test-----------
data = dict(
    samples_per_gpu=1,
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
evaluation = dict(interval=12, pipeline=test_pipeline) 
optimizer = dict(
    type='AdamW',
    lr=2e-4, #---2e-4
    weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    warmup_iters=500, 
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
#----------------------------------------

momentum_config = None
log_config = dict(
    interval=50, #---50
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

total_epochs = 12
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/NewScenes_Final/rcfusion_lss'

load_lift_from = 'work_dirs/NewScenes_Final/LSS/epoch_24.pth'
resume_from = None
load_from = 'work_dirs/NewScenes_Final/radarpillarnet/epoch_24.pth'
workflow = [('train', 1)]

use_old_custom_multi_gpu_test=True
