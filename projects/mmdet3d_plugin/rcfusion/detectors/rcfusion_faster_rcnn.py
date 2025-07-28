import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np
from .BEVCross_modal_attention import Cross_Modal_Fusion
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import MVXFasterRCNN
#-------------更改depthnet------------
#-------------使用bevpoolv2------------
from .cam_stream_lss_bevpoolv2_depthnet import LiftSplatShoot_Depth as LiftSplatShoot
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

@DETECTORS.register_module()
class RCFusion_FasterRCNN(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""
    #---img_depth默认kld
    def __init__(self, freeze_img=False, lss=False, rc_fusion='cross_attention', camera_stream=False,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.6, num_views=6, se=False,
                final_dim=(900, 1600), pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, imc=256, lic=384, norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),**kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(RCFusion_FasterRCNN, self).__init__(**kwargs)
        self.num_views = num_views
        self.rc_fusion = rc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.se = se  #--融合
        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
            pc_range=pc_range,camera_depth_range=camera_depth_range, final_dim=final_dim, downsample=downsample,norm_cfg=norm_cfg)
        if rc_fusion == "concat":
            if se:
                self.seblock = SE_Block(lic)
            self.reduc_conv = ConvModule(
                lic + imc, #--384+256
                lic, #--384
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                inplace=False)
        elif rc_fusion == "cross_attention":
            self.cross_attention = Cross_Modal_Fusion(kernel_size=3,norm_cfg=norm_cfg)
        #-------这里初始化先调用init_weights进行初始化，然后图片分支冻结--
        #-------之后再train文件中读取预训练权重------------
        self.freeze_img = freeze_img
        # self.init_weights(pretrained=kwargs.get('pretrained', None))#--已经在父类中init_cfg
        self.freeze()

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.lift:
                for param in self.lift_splat_shot_vis.parameters():
                    param.requires_grad = False


    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas) #--[torch.Size([6, 256, 112, 200])]
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        if self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN//self.num_views
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W) #--torch.Size([1, 6, 256, 112, 200])
            rots = []
            trans = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                for mat in img_metas[sample_idx]['lidar2img']:  #---['lidar2img']应该是最原始的
                    #-------------4090 cuda11报cusolver的错误，这里改成cpu-------
                    mat = torch.Tensor(mat)
                    rot_list.append(mat.inverse()[:3, :3].to(img_feats_view.device)) #---求逆了img2lidar
                    trans_list.append(mat.inverse()[:3, 3].view(-1).to(img_feats_view.device))
                rot_list = torch.stack(rot_list, dim=0) #--torch.Size([6, 3, 3])
                trans_list = torch.stack(trans_list, dim=0) #--torch.Size([6, 3])
                rots.append(rot_list)
                trans.append(trans_list)
            rots = torch.stack(rots) #---torch.Size([1, 6, 3, 3])
            trans = torch.stack(trans) #--torch.Size([1, 6, 3])
            lidar2img_rt = img_metas[sample_idx]['lidar2img']  #### extrinsic parameters for multi-view images
            #--torch.Size([1, 256, 200, 200])，torch.Size([1, 6, 41, 112, 200])
            img_bev_feat, depth_dist = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas)
            # print(img_bev_feat.shape, pts_feats[-1].shape)
            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.rc_fusion == "concat":
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))] #--[2,384,200,200]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
                elif self.rc_fusion == "cross_attention":
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear',
                                                     align_corners=True)
                    #pts_feats = [self.reduc_radar_conv(pts_feats[0])]
                    pts_feats = [self.cross_attention(img_bev_feat, pts_feats[0])]
        # #         # #-----------绘制一下tensor----
        # from projects.mmdet3d_plugin.models.utils.visual import draw_bev
        # draw_bev(pts_feats[0])
        
        return dict(
            img_feats = img_feats,
            pts_feats = pts_feats,
            depth_dist = depth_dist
        )
        # return (img_feats, pts_feats, depth_dist)
    
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                # pts_feats, img_feats, img_metas, rescale=rescale)
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d)
        img_feats = feature_dict['img_feats'] #---6路图像特征
        pts_feats = feature_dict['pts_feats'] #--bev特征
        depth_dist = feature_dict['depth_dist']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts) #----{'loss_cls': ,'loss_bbox': ,'loss_dir': }
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals) #--空
            if img_depth is not None: 
                loss_depth,min_depth = self.lift_splat_shot_vis.get_depth_loss(depth_labels=img_depth, depth_preds=depth_dist, loss_depth_type=self.img_depth_loss_method) 
                loss_depth = self.img_depth_loss_weight*loss_depth
                losses.update(img_depth_loss=loss_depth)

                
            losses.update(losses_img)
        return losses
    
    def depth_dist_loss(self, predict_depth_dist, gt_depth, loss_method='kld', img=None):
        # predict_depth_dist: B, N, D, H, W
        # gt_depth: B, N, H', W'
        B, N, D, H, W = predict_depth_dist.shape #--[1,6,41,112,200]
        guassian_depth, min_depth = gt_depth[..., 1:], gt_depth[..., 0] #----[1,6,112,200,41] [1,6,112,200]
        mask = (min_depth>=self.camera_depth_range[0]) & (min_depth<=self.camera_depth_range[1]) #--[1,6,112,200]
        mask = mask.view(-1) #--134400
        guassian_depth = guassian_depth.view(-1, D)[mask] #--[69901,41]
        predict_depth_dist = predict_depth_dist.permute(0, 1, 3, 4, 2).reshape(-1, D)[mask] #--[69901,41]
        if loss_method=='kld':
            #----------这里按照openoccupancy加了一个数，reduction='batchmean'，防止出现0导致无法训练nan
            loss = F.kl_div(torch.log(predict_depth_dist + 1e-4), guassian_depth, reduction='batchmean', log_target=False)
        elif loss_method=='mse':
            loss = F.mse_loss(predict_depth_dist, guassian_depth)
        else:
            raise NotImplementedError
        return loss

