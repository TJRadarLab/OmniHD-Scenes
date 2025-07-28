"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
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

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

#----------预测深度概率并将特征与概率相乘，bevpoolv2不进行外积-----
class CamEncode(nn.Module):
    def __init__(self, D, C, inputC):
        super(CamEncode, self).__init__()
        self.D = D #--41
        self.C = C #--64
        self.depthnet = nn.Conv2d(inputC, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # Depth
        x = self.depthnet(x) 

        depth = self.get_depth_dist(x[:, :self.D]) 

        new_x = x[:, self.D:(self.D + self.C),...]
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x) 

        return x, depth


class LiftSplatShoot(nn.Module):
    def __init__(self, lss=False, final_dim=(900, 1600), camera_depth_range=[4.0, 45.0, 1.0], pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, grid=3, inputC=256, camC=64):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            final_dim: actual RGB image size for actual BEV coordinates, default (900, 1600)
            downsample (int): the downsampling rate of the input camera feature spatial dimension (default (224, 400)) to final_dim (900, 1600), default 4. 
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            pc_range: point cloud range.
            inputC: input camera feature channel dimension (default 256).
            grid: stride for splat, see https://github.com/nv-tlabs/lift-splat-shoot.

        """
        super(LiftSplatShoot, self).__init__()
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
        #------------与BaseModule中的init冲突，longtensor无法取mean----------------
        #------------参数会在cuda下，需要修改-----
        # self.dx = nn.Parameter(dx, requires_grad=False)  #---tensor([0.5000, 0.5000, 0.5000], device='cuda:0')--
        # self.bx = nn.Parameter(bx, requires_grad=False) #----tensor([-49.7500, -49.7500,  -4.7500], device='cuda:0')
        # self.nx = nn.Parameter(nx, requires_grad=False) #---tensor([200, 200,  16], device='cuda:0')
        
        self.dx = dx #---tensor([0.5000, 0.5000, 0.5000], device='cuda:0')--
        self.bx = bx#tensor([-59.7500, -39.7500,  -2.7500])
        self.nx = nx #---tensor([240, 160,  16], device='cuda:0')
 
        self.downsample = downsample
        self.fH, self.fW = self.final_dim[0] // self.downsample, self.final_dim[1] // self.downsample
        self.camC = camC
        self.inputC = inputC
        self.frustum = self.create_frustum() #--#--torch.Size([59, 135, 240, 3]),代表原始图像大小和深度
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.inputC)
        

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        z = self.grid_conf['zbound']
        cz = int(self.camC * ((z[1] - z[0]) // z[2]))
        self.lss = lss
        self.bevencode = nn.Sequential(
            nn.Conv2d(cz, cz, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cz),
            nn.ReLU(inplace=True),
            nn.Conv2d(cz, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, inputC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inputC),
            nn.ReLU(inplace=True)
        )
        if self.lss:
          self.bevencode = nn.Sequential(
            nn.Conv2d(cz, camC, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(camC),
            BevEncode(inC=camC, outC=inputC)

        )
    #-在图像平面创建网格，每个代表的尺寸为
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim  #---[1080,1920]
        fH, fW = self.fH, self.fW  #--[135,240]
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) 
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3按xyz順序
        frustum = torch.stack((xs, ys, ds), -1) #--[59,135,240,3]
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, post_rots=None, post_trans=None,extra_rots=None,extra_trans=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  #--torch.Size([1, 6, 3])
        # ADD
        # undo post-transformation
        # B x N x D x H x W x 3
        if post_rots is not None or post_trans is not None:
            if post_trans is not None:
                points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots is not None:
                points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        else:
            points = self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)  # B x N x D x H x W x 3 x 1 torch.Size([1, 6, 69,135,240,3, 1])

        # cam_to_ego rots和trans已经是img2lidar了
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5) #torch.Size([1, 6, 59, 135, 240, 3, 1])
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3) #--torch.Size([1, 6, 59, 135, 240, 3])
        #---转到lidar坐标系下
        if extra_rots is not None or extra_trans is not None:
            if extra_rots is not None:
                points = extra_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            if extra_trans is not None:
                points += extra_trans.view(B, N, 1, 1, 1, 3)
        return points
    #--------这里输出改了，self.camencode不进行外积-----
    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        x, depth = self.camencode(x) #
        for shape_id in range(3):
            assert depth.shape[shape_id+1] == self.frustum.shape[shape_id]
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
        
        feat = feat.permute(0, 1, 3, 4, 2).contiguous() 
        bev_feat_shape = (depth.shape[0], self.nx[2],
                          self.nx[1], self.nx[0],
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # # collapse Z
        # if self.collapse_z:
        #     bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
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
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device) #--B * N * D * H * W
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten() #--B * N * D * H * W
        # convert coordinate into the voxel space
        coor = ((coor - (self.bx.to(coor) - self.dx.to(coor) / 2.)) / self.dx.to(coor)) #--torch.Size([1, 6, 59, 135, 240, 3])
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor) #--torch.Size([11469600, 1])
        coor = torch.cat((coor, batch_idx), 1) #--torch.Size([11469600, 4])

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.nx[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.nx[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.nx[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept] #--torch.Size([2893283, 4])torch.Size([2893283])
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.nx[2].to(coor) * self.nx[1].to(coor) * self.nx[0].to(coor))
        ranks_bev += coor[:, 2] * (self.nx[1].to(coor) * self.nx[0].to(coor))
        ranks_bev += coor[:, 1] * self.nx[0].to(coor) + coor[:, 0]
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
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()
    

    def get_voxels(self, x, rots=None, trans=None, post_rots=None, post_trans=None,extra_rots=None,extra_trans=None):
        geom = self.get_geometry(rots, trans, post_rots, post_trans,extra_rots,extra_trans)#--torch.Size([1, 6, 59, 135, 240, 3])
        
        x, depth = self.get_cam_feats(x) #torch.Size([1, 6, 64, 135, 240])，torch.Size([1, 6, 59, 135, 240])

        x = self.voxel_pooling_v2(geom, depth, x) #改成bevpool
        
        return x, depth

    def s2c(self, x):
        bev = torch.cat(x.unbind(dim=2), 1) #torch.Size([1, 1024, 160, 240])B,CxZ,Y,X
        return bev

    def forward(self, x, rots, trans, lidar2img_rt=None, img_metas=None, post_rots=None, post_trans=None, extra_rots=None,extra_trans=None):
        
        
        x, depth = self.get_voxels(x, rots, trans, post_rots, post_trans,extra_rots,extra_trans) # [B, C, Z, X, Y]torch.Size([1, 64, 16, 288, 224])
        
        bev = self.s2c(x)
        x = self.bevencode(bev) #--torch.Size([1, 256, 160, 240])
        return x, depth

