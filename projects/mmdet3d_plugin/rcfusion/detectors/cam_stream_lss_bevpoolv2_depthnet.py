"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.runner import force_fp32
import torch
from torch import nn
from torchvision.models.resnet import resnet18
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from torchvision.utils import save_image
from mmdet3d.models.fusion_layers import apply_3d_transformation
import torch.nn.functional as F
from projects.mmdet3d_plugin.ops.bev_pool_v2.bev_pool import bev_pool_v2
from projects.mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmcv.cnn import build_conv_layer, build_norm_layer
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.shape[2:],  mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(kept)

        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class CamEncode(nn.Module):
    def __init__(self, D, C, inputC, norm_cfg):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.depthnet = DepthNet(in_channels=inputC, mid_channels=inputC, context_channels=C, depth_channels=D, norm_cfg=norm_cfg)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])

        new_x = x[:, self.D:(self.D + self.C), ...]
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x, depth


class LiftSplatShoot_Depth(nn.Module):
    def __init__(self, lss=False, final_dim=(900, 1600), camera_depth_range=[4.0, 45.0, 1.0], pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, grid=3, inputC=256, camC=64, norm_cfg=None):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            final_dim: actual RGB image size for actual BEV coordinates, default (900, 1600)
            downsample (int): the downsampling rate of the input camera feature spatial dimension (default (224, 400)) to final_dim (900, 1600), default 4.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision which is not mentioned in paper.
            pc_range: point cloud range.
            inputC: input camera feature channel dimension (default 256).
            grid: stride for splat, see https://github.com/nv-tlabs/lift-splat-shoot.

        """
        super(LiftSplatShoot_Depth, self).__init__()
        self.pc_range = pc_range
        self.grid_conf = {
            'xbound': [pc_range[0], pc_range[3], grid],
            'ybound': [pc_range[1], pc_range[4], grid],
            'zbound': [pc_range[2], pc_range[5], grid],
            'dbound': camera_depth_range,
        }
        self.final_dim = final_dim
        self.grid = grid

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'], )

        self.dx = torch.tensor(dx.clone().detach().numpy(), requires_grad=False)
        self.bx = torch.tensor(bx.clone().detach().numpy(), requires_grad=False)
        self.nx = torch.tensor(nx.clone().detach().numpy(), requires_grad=False)
        self.downsample = downsample
        self.fH, self.fW = self.final_dim[0] // self.downsample, self.final_dim[1] // self.downsample
        self.camC = camC
        self.inputC = inputC
        self.norm_cfg = norm_cfg
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.inputC, self.norm_cfg)
        self.constant_std = 0.5
        self.camera_depth_range = camera_depth_range

        self.use_quickcumsum = True
        z = self.grid_conf['zbound']
        cz = int(self.camC * ((z[1] - z[0]) // z[2]))
        self.lss = lss
        self.bevencode = nn.Sequential(
            nn.Conv2d(cz, cz, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, cz)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(cz, 512, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 512)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 512)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(512, inputC, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, inputC)[1],
            nn.ReLU(inplace=True)
        )
        if self.lss:
            self.bevencode = nn.Sequential(
                nn.Conv2d(cz, camC, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, camC)[1],
                BevEncode(inC=camC, outC=inputC)
            )

    def create_frustum(self):
        ogfH, ogfW = self.final_dim
        fH, fW = self.fH, self.fW
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, post_rots=None, post_trans=None, extra_rots=None, extra_trans=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        if post_rots is not None or post_trans is not None:
            if post_trans is not None:
                points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots is not None:
                points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        else:
            points = self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)

        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        if extra_rots is not None or extra_trans is not None:
            if extra_rots is not None:
                points = extra_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            if extra_trans is not None:
                points += extra_trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        x, depth = self.camencode(x)
        for shape_id in range(3):
            assert depth.shape[shape_id + 1] == self.frustum.shape[shape_id]
        x = x.view(B, N, self.camC, H, W)
        depth = depth.view(B, N, self.D, H, W)
        return x, depth

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            return None

        nx = self.nx.to(coor.device)
        feat = feat.permute(0, 1, 3, 4, 2).contiguous()
        bev_feat_shape = (depth.shape[0], nx[2],
                          nx[1], nx[0],
                          feat.shape[-1])
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        dx = self.dx.to(coor.device)
        nx = self.nx.to(coor.device)
        bx = self.bx.to(coor.device)
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        coor = ((coor - (bx - dx / 2.)) / dx)
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        kept = (coor[:, 0] >= 0) & (coor[:, 0] < nx[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < nx[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < nx[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        ranks_bev = coor[:, 3] * (
            nx[2] * nx[1] * nx[0])
        ranks_bev += coor[:, 2] * (nx[1] * nx[0])
        ranks_bev += coor[:, 1] * nx[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(), ranks_feat.int().contiguous(), interval_starts.int().contiguous(), interval_lengths.int().contiguous()

    def get_voxels(self, x, rots=None, trans=None, post_rots=None, post_trans=None, extra_rots=None, extra_trans=None):
        geom = self.get_geometry(rots, trans, post_rots, post_trans, extra_rots, extra_trans)
        x, depth = self.get_cam_feats(x)

        x = self.voxel_pooling_v2(geom, depth, x)

        return x, depth

    def s2c(self, x):
        bev = torch.cat(x.unbind(dim=2), 1)
        return bev

    def forward(self, x, rots, trans, lidar2img_rt=None, img_metas=None, post_rots=None, post_trans=None, extra_rots=None, extra_trans=None):

        x, depth = self.get_voxels(x, rots, trans, post_rots, post_trans, extra_rots, extra_trans)

        bev = self.s2c(x)
        x = self.bevencode(bev)
        return x, depth

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]

        return gt_depths_vals, gt_depths.float()

    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())

        return depth_loss

    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.camera_depth_range, constant_std=self.constant_std)
        depth_values_return = depth_values.clone()
        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.camera_depth_range[0]) & (depth_values <= (self.camera_depth_range[1] - self.camera_depth_range[2]))

        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)[fg_mask]

        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)

        return depth_loss, depth_values_return

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds, loss_depth_type):
        if loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)

        elif loss_depth_type == 'kld':
            depth_loss, min_depth = self.get_klv_depth_loss(depth_labels, depth_preds)

        else:
            pdb.set_trace()

        return depth_loss, min_depth



class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, norm_cfg=dict(type='BN2d')):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=build_norm_layer(norm_cfg, mid_channels)[1])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, norm_cfg=None):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                context_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            ASPP(mid_channels, mid_channels, norm_cfg=norm_cfg),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x):

        x = self.reduce_conv(x)
        context = self.context_conv(x)
        depth = self.depth_conv(x)

        return torch.cat([depth, context], dim=1)