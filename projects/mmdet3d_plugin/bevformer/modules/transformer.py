# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
    #-------初始化层包括levelemb，camemb，参考点，canbus，----
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims)) #--(4,256)
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)) #--(6,256)
        self.reference_points = nn.Linear(self.embed_dims, 3) #---用来生成obj_q参考点--
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        ##这里会先逐层参数进行xavier初始化，只考虑大于一维的weight，bias一般不考虑--
        #----所有的模块都会循环一遍，然后for m in self.modules():会再调用其本身的初始化
        for p in self.parameters(): 
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        #----PerceptionTransformer,encoder,decoder,reference_points.can_bus_mlp
        #-----并且逐层迭代，内部的linear等层已经在下列层的init_weights中初始化-------
        for m in self.modules():  
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()  #---调用的这个---
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        #---这个地方没调用这个，nn.Sequential没有weights和bias属性，这里不管了，
        #----for p in self.parameters():中对weights初始化过一次了 ------
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.) 

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)  #--（1,6,256,23,40）
        #-----这里注意传进来的时候是nn.Parameter,requires_grad是true，经过---
        #-----下面操作就变成一个tensor了，requires_grad是false----
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) #--重复batch维，（22500，1,256）
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) #---(22500,1,256)

    #---------------------这里对于历史BEV的操作还要再细节看-------------------------
        # obtain rotation angle and shift with ego motion
        #----通过ego pose算得的与前一帧的位置差-----
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        #--------全局坐标系下yaw角度，332.57---------
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0] #---0.682666
        grid_length_x = grid_length[1] #---0.682666
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        # bev_angle = ego_angle - translation_angle
        bev_angle = translation_angle - ego_angle #---修改
        shift_y = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_y / bev_h  #---车头前x左y
        shift_x = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        # shift_y = translation_length * \
        #     np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h  #--车头朝前是y
        # shift_x = translation_length * \
        #     np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w #--车头右是x
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy (2,1)-->(1,2)
    #-----历史BEV旋转--------
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1] #---后帧与前帧的yaw差
    
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    
                    # from projects.mmdet3d_plugin.models.utils.visual import draw_bev
                    # draw_bev(tmp_prev_bev,'bevfore_rot')
                   
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center) #----这里应该根据bev中心点--
                   
                    # draw_bev(tmp_prev_bev,'after_rot')

                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
    #-------------------------------------------------------------------
        # add can bus signals canbus嵌入---------
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]  #--（1,1,256）
        bev_queries = bev_queries + can_bus * self.use_can_bus #--(22500,1,256)

        feat_flatten = []
        spatial_shapes = []
        #----这里的特征加入了相机嵌入和多尺度特征层嵌入，没明白啥意思都是初始化的变量----
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape  #--[1,6,256,23,40]
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) #--[6,1,920,256]
            if self.use_cams_embeds: #--[6,1,1,256]
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) #--[6,1,920,256]
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device) #--torch.int64 tensor([[23,40]])
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) #--torch.int64 tensor([0])

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims) [6,920,1,256]
        #-----这里进入BEVFormerEncoder进行编码---
        bev_embed = self.encoder(
            bev_queries,  #----[22500,1,256]
            feat_flatten, #----[6,920,1,256]
            feat_flatten,#----[6,920,1,256]
            bev_h=bev_h, #150
            bev_w=bev_w,#150
            bev_pos=bev_pos,#----[22500,1,256]
            spatial_shapes=spatial_shapes, #--tensor([[23,40]])
            level_start_index=level_start_index, #--tenosr[0]
            prev_bev=prev_bev,
            shift=shift, #--tensor([[shift_x, shift_y]])
            **kwargs
        )

        return bev_embed #--[1,22500,256]

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        #--torch.Size([1, 22500, 256])
        bev_embed = self.get_bev_features(
            mlvl_feats, #--torch.Size([1, 6, 256, 23, 40])
            bev_queries, #--nn.parameter,torch.Size([22500, 256])
            bev_h, #--150
            bev_w, #--150
            grid_length=grid_length, #--(0.682666,0.682666)
            bev_pos=bev_pos, #--torch.Size([1, 256, 150, 150])
            prev_bev=prev_bev, #--[1,22500,256]
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        #---可视化最终bev------

        # final_bev = bev_embed[0].reshape(bev_h,bev_w,-1).permute(2, 0, 1)
        # from projects.mmdet3d_plugin.models.utils.visual import draw_bev
        # draw_bev(final_bev,'final_bevformer')
        
        #----下面是obj q----
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1) #---[900,256][900,256]
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)#---[1,900,256]
        query = query.unsqueeze(0).expand(bs, -1, -1)#---[1,900,256]
        #-----------用linear生成参考点------
        reference_points = self.reference_points(query_pos)#---[1,900,3]
        reference_points = reference_points.sigmoid() #---映射到0到1，和不一定为1
        init_reference_out = reference_points #----------初始参考点#---[1,900,3]

        query = query.permute(1, 0, 2) #--[900,1,256]
        query_pos = query_pos.permute(1, 0, 2)#--[900,1,256]
        bev_embed = bev_embed.permute(1, 0, 2)#--#--torch.Size([22500, 1,256])
        #-------进入'DetectionTransformerDecoder'------
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches, #--None
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs) #---torch.Size([6, 900, 1, 256]),torch.Size([6, 1, 900, 3])

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
