# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Written by [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

#----video_test_mode=False, in BEVFormer code set
# backbone R50-----
# smaller BEV grid: 180*140
# less encoder layers: 6 -> 3
# smaller input size: 1920*1080 -> (1920*1080)*0.5
# multi-scale feautres -> single scale features (C5)


point_cloud_range = [-60, -40, -3.0, 60, 40, 5.0] #---Ego-vehicle / LiDAR coordinate frame
voxel_size = [0.25, 0.25, 8]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) 
# For newScenes we usually do 9-class detection
class_names = [
    'car', 'pedestrian', 'rider', 'large_vehicle']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False)

_dim_ = 256 
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1  
bev_h_ = 160  #--ymax - ymin
bev_w_ = 240  #--xmax - xmin
queue_length = 3 # each sequence contains `queue_length` frames.

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True, #----Use this to control single-frame mode
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3,), 
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True, # using checkpoint to save GPU memory
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_, ##256
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_, #1
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_, #--160
        bev_w=bev_w_,#--240
        num_query=900,
        num_classes=4, #----This is 4 classes
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,  
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            rotate_center=[80,120], #-----This is the BEV center point
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_, #--256
                            num_levels=1), #--single feature map
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_, #256
                                num_points=8,
                                num_levels=_num_levels_), #1
                            embed_dims=_dim_, #256
                        )
                    ],
                    feedforward_channels=_ffn_dim_, #--512
                    ffn_dropout=0.1, #---Initialized in MyCustomBaseTransformerLayer
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(  #-------DETR3D--------
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True, 
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer', 
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,#--512
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-70, -50, -10.0, 70, 50, 10.0],
            pc_range=point_cloud_range,
            max_num=300, #---topk=300
            voxel_size=voxel_size,
            num_classes=4), 
        positional_encoding=dict(
            type='LearnedPositionalEncoding', 
            num_feats=_pos_dim_, #128
            row_num_embed=bev_h_, 
            col_num_embed=bev_w_, 
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True, 
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)), 
    # model training and testing settings
    train_cfg=dict(pts=dict( 
        grid_size=[0, 0, 1],
        voxel_size=voxel_size, 
        point_cloud_range=point_cloud_range,
        out_size_factor=4, 
        assigner=dict(
            type='HungarianAssigner3D',     
            cls_cost=dict(type='FocalLossCost', weight=2.0),#-----in core/bbox/match_costs
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), 
            pc_range=point_cloud_range))))

dataset_type = 'CustomNewScenesDataset' 
data_root = 'data/NewScenes_Final/' 
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_newsc', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg), 
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_newsc', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080), 
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

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
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'newscenes-final_infos_temporal_val.pkl',
             pipeline=test_pipeline, 
             bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'newscenes-final_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)


evaluation = dict(interval=4, pipeline=test_pipeline)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01) 

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
momentum_config = None
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


total_epochs = 24
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/NewScenes_Final/bevformer_T_R101'

load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
resume_from = None
workflow = [('train', 1)]

use_old_custom_multi_gpu_test=True

