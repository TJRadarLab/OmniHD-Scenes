# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Written by [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models import builder
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import HEADS, build_loss
#------使用resnet+fpn--------------
from .bev_encoder import BevEncode
#-----使用简单的卷积---------
# from .bev_encoder_small import BevEncode
from .map_head import BevFeatureSlicer
from mmcv.runner import auto_fp16, force_fp32

import pdb


@HEADS.register_module()
class MultiTaskHead(BaseModule):
    def __init__(self, init_cfg=None, 
                 in_channels=64,
                 out_channels=256,
                 bev_encode_block='Basic',
                 bev_encoder_type='resnet18',
                 bev_encode_depth=[1, 1, 1],
                 num_channels=None,
                 backbone_output_ids=None,
                 norm_cfg=dict(type='BN'),
                 bev_encoder_fpn_type='lssfpn',
                 grid_conf=None,
                 det_grid_conf=None,
                 occ_grid_conf=None, 
                 task_enbale=None, 
                 task_weights=None,
                 out_with_activision=False,
                 shared_feature=False,
                 cfg_3dod=None,
                 cfg_occ=None,

                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(MultiTaskHead, self).__init__(init_cfg)

        self.fp16_enabled = False
        self.task_enbale = task_enbale
        self.task_weights = task_weights


        if det_grid_conf is None:
            det_grid_conf = grid_conf

        # build task-features
        self.task_names_ordered = ['3dod', 'occ']
        self.taskfeat_encoders = nn.ModuleDict()
        assert bev_encoder_type == 'resnet18'

        # whether to use shared features
       
        self.shared_feature = shared_feature
        if self.shared_feature:
            self.taskfeat_encoders['shared'] = BevEncode(
                numC_input=in_channels,
                numC_output=out_channels,
                num_channels=num_channels,
                backbone_output_ids=backbone_output_ids,
                num_layer=bev_encode_depth,
                bev_encode_block=bev_encode_block,
                norm_cfg=norm_cfg,
                bev_encoder_fpn_type=bev_encoder_fpn_type,
                out_with_activision=out_with_activision,
            )
        else:
            for task_name in self.task_names_ordered:
                is_enable = task_enbale.get(task_name, False)
                if not is_enable:
                    continue

                self.taskfeat_encoders[task_name] = BevEncode(
                    numC_input=in_channels,
                    numC_output=out_channels,
                    num_channels=num_channels,
                    backbone_output_ids=backbone_output_ids,
                    num_layer=bev_encode_depth,
                    bev_encode_block=bev_encode_block,
                    norm_cfg=norm_cfg,
                    bev_encoder_fpn_type=bev_encoder_fpn_type,
                    out_with_activision=out_with_activision,
                )
        #-------------------------------------------------






        # build task-decoders
        self.task_decoders = nn.ModuleDict()
        self.task_feat_cropper = nn.ModuleDict()

        # 3D object detection
        if task_enbale.get('3dod', False):
            cfg_3dod.update(train_cfg=train_cfg)
            cfg_3dod.update(test_cfg=test_cfg)

            self.task_feat_cropper['3dod'] = BevFeatureSlicer(
                grid_conf, det_grid_conf)
            self.task_decoders['3dod'] = builder.build_head(cfg_3dod)

        # occupancy
        if task_enbale.get('occ', False):

            self.task_feat_cropper['occ'] = BevFeatureSlicer(
                grid_conf, occ_grid_conf)
            self.task_decoders['occ'] = builder.build_head(cfg_occ)


    def init_weights(self):
        if self.task_enbale.get('3dod', False):
            self.task_decoders['3dod'].init_weights()
        if self.task_enbale.get('occ', False):
            self.task_decoders['occ'].init_weights()

#--------------对每个任务中的子loss加权-----------
    def scale_task_losses(self, task_name, task_loss_dict):
        task_sum = 0
        if task_name == '3dod':
            for key, val in task_loss_dict.items():
                task_sum += val[0].item()  #---这里OD是一个[tensor(X)],OCC是tensor(X)
                task_loss_dict[key] = val[0] * self.task_weights.get(task_name, 1.0)
        if task_name == 'occ':
            for key, val in task_loss_dict.items():
                task_sum += val.item()  #---这里OD是一个[tensor(X)],OCC是tensor(X)
                task_loss_dict[key] = val * self.task_weights.get(task_name, 1.0)
        
        
        #--这里在basedetector中时没有计算，只计算key中带有'loss'的
        task_loss_summation = sum(list(task_loss_dict.values()))
        task_loss_dict['{}_sum'.format(task_name)] = task_loss_summation

        return task_loss_dict

    def loss(self, predictions, img_metas, targets):
        loss_dict = {}

        #------Anchor3D------
        if self.task_enbale.get('3dod', False):
            outs = predictions['3dod']
            od_loss_inputs = outs + (targets['gt_bboxes_3d'], targets['gt_labels_3d'], img_metas)
            det_loss_dict = self.task_decoders['3dod'].loss(
                *od_loss_inputs,
                gt_bboxes_ignore=targets['gt_bboxes_ignore']
            )
            loss_dict.update(self.scale_task_losses(
                task_name='3dod', task_loss_dict=det_loss_dict))

        if self.task_enbale.get('occ', False):
            occ_loss_dict = self.task_decoders['occ'].loss(
                predictions['occ'], targets['gt_occ']
            )
            loss_dict.update(self.scale_task_losses(
                task_name='occ', task_loss_dict=occ_loss_dict))


        return loss_dict

#------输出解码的bbox和原始的occ_pred--------------
    def inference(self, predictions, img_metas, rescale):
        res = {}
        # 输出预测的bbox
        if self.task_enbale.get('3dod', False):
            res['bbox_list'] = self.task_decoders['3dod'].get_bboxes(
                *predictions['3dod'],
                img_metas,
                rescale=rescale
            )

        # occ head 输出原始预测#--torch.Size([1, 240, 160, 16, 19])
        if self.task_enbale.get('occ', False):
            res['occ_pred'] = predictions['occ']

        return res
    
#-------------------暂时不用-----------------------
    def forward_with_shared_features(self, bev_feats, targets=None):
        predictions = {}
       

        bev_feats = self.taskfeat_encoders['shared']([bev_feats])

        for task_name in self.task_feat_cropper:
            # crop feature before the encoder
            task_feat = self.task_feat_cropper[task_name](bev_feats)
            
            # task-specific decoder


            task_pred = self.task_decoders[task_name]([task_feat])
            
            predictions[task_name] = task_pred

        return predictions
#--------------------------------------------------------------------

#-----------------输出dict{'3dod':([cls_scoretensor],[bbox_predtensor],[dir_cls_predstensor]),'occ':tensor}
    def forward(self, bev_feats, targets=None): #--torch.Size([1, 256, 160, 240])
        if self.shared_feature:
            return self.forward_with_shared_features(bev_feats, targets)

        predictions = {}
        for task_name, task_feat_encoder in self.taskfeat_encoders.items():

            # crop feature before the encoder [torch.Size([1, 256, 160, 240])]
            task_feat = self.task_feat_cropper[task_name](bev_feats)

            # task-specific feature encoder
            task_feat = task_feat_encoder(task_feat) #torch.Size([1, 256, 160, 240])

            # task-specific decoder
            
            task_pred = self.task_decoders[task_name]([task_feat])

            
            predictions[task_name] = task_pred

        return predictions
