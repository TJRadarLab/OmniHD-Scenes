# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
import torch.nn.functional as F


@HEADS.register_module()
class BEVOCCHead3D(BaseModule):
    def __init__(self,
                 in_dim=32,
                 out_dim=32,
                 num_classes=18,
                 use_predicter=True,
                 loss_occ=None
                 ):
        super(BEVOCCHead3D, self).__init__()
        self.out_dim = 32
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)

    def forward(self, occ_feats):
        """
        Args:
            occ_feats: (B, C, Dx, Dy, Dz)

        Returns:

        """
        # (B, C, Dx, Dy, Dz) --> (B, C, Dx, Dy, Dz) --> (B, Dx, Dy, Dz, C)
        occ_pred = self.final_conv(occ_feats).permute(0, 2, 3, 4, 1)
        if self.use_predicter:
            # (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, 2*C) --> (B, Dx, Dy, Dz, n_cls)
            occ_pred = self.predicter(occ_pred) # torch.Size([1, 200, 200, 16, 18])

        return occ_pred

    def loss(self, occ_pred, gt_occ):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = gt_occ.long()
        loss_ssc = self.sem_scal_loss(occ_pred, voxel_semantics.long()) \
                    + self.geo_scal_loss(occ_pred, voxel_semantics.long())
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = occ_pred.reshape(-1, self.num_classes)
        loss_occ = self.loss_occ(preds, voxel_semantics)

        loss['loss_ssc'] = loss_ssc
        loss['loss_occ'] = loss_occ
        
        # total_loss = loss_occ + loss_ssc
        # loss['loss_occ'] = total_loss   # 修改loss为总loss


        return loss
    
    def geo_scal_loss(self, preds, ssc_target, semantic=True):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        if semantic:
            pred = F.softmax(pred, dim=1)

            # Compute empty and nonempty probabilities
            empty_probs = pred[:, 0, :, :, :]
        else:
            empty_probs = 1 - torch.sigmoid(pred)
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != 0  # 迁移过来occ3d中17代表空 原为0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )
    
    def sem_scal_loss(self, preds, ssc_target):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)   # torch.Size([1, 17, 25, 25, 2])
        loss = 0
        count = 0
        mask = ssc_target != 255    # 剔除255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] ## 原surroundocc适配格式
            # p = pred[:, :, :, :, i] ## 适配bevformer_occ格式

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count
    

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class TPVOccHead3D(BaseModule):
    def __init__(self,
                 in_dim=32,
                 hidden_dims=64,
                 out_dim=32,
                 num_classes=18,
                 use_predicter=True,
                 loss_occ=None
                 ):
        super(TPVOccHead3D, self).__init__()
        self.tpv_h = 200
        self.tpv_w = 200
        self.tpv_z = 16
        self.in_dim = in_dim
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        # self.final_conv = ConvModule(
        #     in_dim,
        #     out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=True,
        #     conv_cfg=dict(type='Conv3d')
        # )
        self.use_predicter = use_predicter

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.in_dim, hidden_dims),
        #     nn.Softplus(),
        #     nn.Linear(hidden_dims, self.out_dim)
        # )

        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)

    def forward(self, tpv_list):
        """
        Args:
            tpv_list: (B, C, Dx, Dy, Dz)

        Returns:
        """

        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.tpv_z)
        tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.tpv_w, -1, -1)
        tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.tpv_h, -1)

        fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2).permute(0, 2, 1)   # torch.Size([1, 256, 80000])
    
        if self.use_predicter:
            occ_pred = self.predicter(fused_vox)

        # occ_pred = occ_pred.permute(0, 2, 1)
        occ_pred = occ_pred.reshape(bs, self.tpv_h, self.tpv_w, self.tpv_z, self.num_classes)
        return occ_pred

    def loss(self, occ_pred, gt_occ):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = gt_occ.long()
        loss_ssc = self.sem_scal_loss(occ_pred, voxel_semantics.long()) \
                    + self.geo_scal_loss(occ_pred, voxel_semantics.long())
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = occ_pred.reshape(-1, self.num_classes)
        loss_occ = self.loss_occ(preds, voxel_semantics)

        loss['loss_ssc'] = loss_ssc
        loss['loss_occ'] = loss_occ
        
        # total_loss = loss_occ + loss_ssc
        # loss['loss_occ'] = total_loss   # 修改loss为总loss


        return loss
    
    def geo_scal_loss(self, preds, ssc_target, semantic=True):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        if semantic:
            pred = F.softmax(pred, dim=1)

            # Compute empty and nonempty probabilities
            empty_probs = pred[:, 0, :, :, :]
        else:
            empty_probs = 1 - torch.sigmoid(pred)
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != 0  # 迁移过来occ3d中17代表空 原为0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )
    
    def sem_scal_loss(self, preds, ssc_target):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)   # torch.Size([1, 17, 25, 25, 2])
        loss = 0
        count = 0
        mask = ssc_target != 255    # 剔除255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] ## 原surroundocc适配格式
            # p = pred[:, :, :, :, i] ## 适配bevformer_occ格式

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count
    

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)
@HEADS.register_module()
class TPVOccHead3Dv2(BaseModule):
    def __init__(self,
                 in_dim=32,
                #  hidden_dims=64,
                 out_dim=32,
                 num_classes=18,
                 use_predicter=True,
                 loss_occ=None
                 ):
        super(TPVOccHead3Dv2, self).__init__()
        self.tpv_h = 200
        self.tpv_w = 200
        self.tpv_z = 16
        self.in_dim = in_dim
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        # self.final_conv = ConvModule(
        #     in_dim,
        #     out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=True,
        #     conv_cfg=dict(type='Conv3d')
        # )
        self.use_predicter = use_predicter

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.in_dim, hidden_dims),
        #     nn.Softplus(),
        #     nn.Linear(hidden_dims, self.out_dim)
        # )

        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.in_dim, self.out_dim),
                nn.Softplus(),
                nn.Linear(self.out_dim, num_classes),
            )

        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)

    def forward(self, tpv_list):
        """
        Args:
            tpv_list: (B, C, Dx, Dy, Dz)

        Returns:
        """

        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.tpv_z)
        tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.tpv_w, -1, -1)
        tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.tpv_h, -1)

        fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2).permute(0, 2, 1)   # torch.Size([1, 256, 80000])
    
        if self.use_predicter:
            occ_pred = self.predicter(fused_vox)

        # occ_pred = occ_pred.permute(0, 2, 1)
        occ_pred = occ_pred.reshape(bs, self.tpv_h, self.tpv_w, self.tpv_z, self.num_classes)
        return occ_pred

    def loss(self, occ_pred, gt_occ):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = gt_occ.long()
        loss_ssc = self.sem_scal_loss(occ_pred, voxel_semantics.long()) \
                    + self.geo_scal_loss(occ_pred, voxel_semantics.long())
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = occ_pred.reshape(-1, self.num_classes)
        loss_occ = self.loss_occ(preds, voxel_semantics)

        loss['loss_ssc'] = loss_ssc
        loss['loss_occ'] = loss_occ
        
        # total_loss = loss_occ + loss_ssc
        # loss['loss_occ'] = total_loss   # 修改loss为总loss


        return loss
    
    def geo_scal_loss(self, preds, ssc_target, semantic=True):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        if semantic:
            pred = F.softmax(pred, dim=1)

            # Compute empty and nonempty probabilities
            empty_probs = pred[:, 0, :, :, :]
        else:
            empty_probs = 1 - torch.sigmoid(pred)
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != 0  # 迁移过来occ3d中17代表空 原为0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )
    
    def sem_scal_loss(self, preds, ssc_target):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)   # torch.Size([1, 17, 25, 25, 2])
        loss = 0
        count = 0
        mask = ssc_target != 255    # 剔除255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] ## 原surroundocc适配格式
            # p = pred[:, :, :, :, i] ## 适配bevformer_occ格式

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count
    

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead3Dv2(BaseModule):
    def __init__(self,
                 in_dim=32,
                 out_dim=32,
                 num_classes=18,
                 use_predicter=True,
                 loss_occ=None
                 ):
        super(BEVOCCHead3Dv2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 32
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.in_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)

    def forward(self, occ_feats):
        """
        Args:
            occ_feats: (B, C, Dx, Dy, Dz)

        Returns:

        """
        # (B, C, Dx, Dy, Dz) --> (B, C, Dx, Dy, Dz) --> (B, Dx, Dy, Dz, C)
        # occ_pred = self.final_conv(occ_feats).permute(0, 2, 3, 4, 1)
        occ_pred = occ_feats.permute(0, 2, 3, 4, 1)
        if self.use_predicter:
            # (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, 2*C) --> (B, Dx, Dy, Dz, n_cls)
            occ_pred = self.predicter(occ_pred) # torch.Size([1, 200, 200, 16, 18])

        return occ_pred

    def loss(self, occ_pred, gt_occ):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = gt_occ.long()
        loss_ssc = self.sem_scal_loss(occ_pred, voxel_semantics.long()) \
                    + self.geo_scal_loss(occ_pred, voxel_semantics.long())
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = occ_pred.reshape(-1, self.num_classes)
        loss_occ = self.loss_occ(preds, voxel_semantics)

        loss['loss_ssc'] = loss_ssc
        loss['loss_occ'] = loss_occ
        
        # total_loss = loss_occ + loss_ssc
        # loss['loss_occ'] = total_loss   # 修改loss为总loss


        return loss
    
    def geo_scal_loss(self, preds, ssc_target, semantic=True):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        if semantic:
            pred = F.softmax(pred, dim=1)

            # Compute empty and nonempty probabilities
            empty_probs = pred[:, 0, :, :, :]
        else:
            empty_probs = 1 - torch.sigmoid(pred)
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != 0  # 迁移过来occ3d中17代表空 原为0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )
    
    def sem_scal_loss(self, preds, ssc_target):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)   # torch.Size([1, 17, 25, 25, 2])
        loss = 0
        count = 0
        mask = ssc_target != 255    # 剔除255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] ## 原surroundocc适配格式
            # p = pred[:, :, :, :, i] ## 适配bevformer_occ格式

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count
    

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)



@HEADS.register_module()
class BEVOCCHead2Dv2(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=False,
                 num_classes=19,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None
                 ):
        super(BEVOCCHead2Dv2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes*Dz),
            )
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        # if self.class_balance:
        #     class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
        #     self.cls_weights = class_weights
        #     loss_occ['class_weight'] = class_weights        # ce loss
        self.loss_occ = build_loss(loss_occ)

    def forward(self, occ_feats):
        """
        Args:
            occ_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C) torch.Size([1, 240, 160, 256])
        occ_pred = self.final_conv(occ_feats[0]).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred) # torch.Size([1, 240, 160, 304])
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes) #--torch.Size([1, 240, 160, 16, 19])
        return occ_pred

    def loss(self, occ_pred, gt_occ):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = gt_occ.long()
        loss_ssc = self.sem_scal_loss(occ_pred, voxel_semantics.long()) \
                    + self.geo_scal_loss(occ_pred, voxel_semantics.long())
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = occ_pred.reshape(-1, self.num_classes)
        loss_occ = self.loss_occ(preds, voxel_semantics)

        loss['loss_ssc'] = loss_ssc
        loss['loss_occ'] = loss_occ
        
        # total_loss = loss_occ + loss_ssc
        # loss['loss_occ'] = total_loss   # 修改loss为总loss


        return loss
    
    def geo_scal_loss(self, preds, ssc_target, semantic=True):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        if semantic:
            pred = F.softmax(pred, dim=1)

            # Compute empty and nonempty probabilities
            empty_probs = pred[:, 0, :, :, :]
        else:
            empty_probs = 1 - torch.sigmoid(pred)
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != 0  # 迁移过来occ3d中17代表空 原为0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )
    
    def sem_scal_loss(self, preds, ssc_target):
        pred = preds.clone().permute(0, 4, 1, 2, 3)
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)   # torch.Size([1, 17, 25, 25, 2])
        loss = 0
        count = 0
        mask = ssc_target != 255    # 剔除255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] ## 原surroundocc适配格式
            # p = pred[:, :, :, :, i] ## 适配bevformer_occ格式

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count
    

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)



# import torch
# from torch import nn
# from mmcv.cnn import ConvModule
# from mmcv.runner import BaseModule
# import numpy as np
# from mmdet3d.models.builder import HEADS, build_loss
# from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
# from ..losses.lovasz_softmax import lovasz_softmax




# @HEADS.register_module()
# class BEVOCCHead3D(BaseModule):
#     def __init__(self,
#                  in_dim=32,
#                  out_dim=32,
#                  use_mask=True,
#                  num_classes=18,
#                  use_predicter=True,
#                  class_balance=False,
#                  loss_occ=None
#                  ):
#         super(BEVOCCHead3D, self).__init__()
#         self.out_dim = 32
#         out_channels = out_dim if use_predicter else num_classes
#         self.final_conv = ConvModule(
#             in_dim,
#             out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=True,
#             conv_cfg=dict(type='Conv3d')
#         )
#         self.use_predicter = use_predicter
#         if use_predicter:
#             self.predicter = nn.Sequential(
#                 nn.Linear(self.out_dim, self.out_dim*2),
#                 nn.Softplus(),
#                 nn.Linear(self.out_dim*2, num_classes),
#             )

#         self.num_classes = num_classes
#         self.use_mask = use_mask
#         self.class_balance = class_balance
#         if self.class_balance:
#             class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
#             self.cls_weights = class_weights
#             loss_occ['class_weight'] = class_weights

#         self.loss_occ = build_loss(loss_occ)

#     def forward(self, img_feats):
#         """
#         Args:
#             img_feats: (B, C, Dz, Dy, Dx)

#         Returns:

#         """
#         # (B, C, Dz, Dy, Dx) --> (B, C, Dz, Dy, Dx) --> (B, Dx, Dy, Dz, C)
#         occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)
#         if self.use_predicter:
#             # (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, 2*C) --> (B, Dx, Dy, Dz, n_cls)
#             occ_pred = self.predicter(occ_pred)

#         return occ_pred

#     def loss(self, occ_pred, voxel_semantics, mask_camera):
#         """
#         Args:
#             occ_pred: (B, Dx, Dy, Dz, n_cls)
#             voxel_semantics: (B, Dx, Dy, Dz)
#             mask_camera: (B, Dx, Dy, Dz)
#         Returns:

#         """
#         loss = dict()
#         voxel_semantics = voxel_semantics.long()
#         if self.use_mask:
#             mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
#             # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
#             voxel_semantics = voxel_semantics.reshape(-1)
#             # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
#             preds = occ_pred.reshape(-1, self.num_classes)
#             # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
#             mask_camera = mask_camera.reshape(-1)

#             if self.class_balance:
#                 valid_voxels = voxel_semantics[mask_camera.bool()]
#                 num_total_samples = 0
#                 for i in range(self.num_classes):
#                     num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
#             else:
#                 num_total_samples = mask_camera.sum()

#             loss_occ = self.loss_occ(
#                 preds,      # (B*Dx*Dy*Dz, n_cls)
#                 voxel_semantics,    # (B*Dx*Dy*Dz, )
#                 mask_camera,        # (B*Dx*Dy*Dz, )
#                 avg_factor=num_total_samples
#             )
#         else:
#             voxel_semantics = voxel_semantics.reshape(-1)
#             preds = occ_pred.reshape(-1, self.num_classes)

#             if self.class_balance:
#                 num_total_samples = 0
#                 for i in range(self.num_classes):
#                     num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
#             else:
#                 num_total_samples = len(voxel_semantics)

#             loss_occ = self.loss_occ(
#                 preds,
#                 voxel_semantics,
#                 avg_factor=num_total_samples
#             )

#         loss['loss_occ'] = loss_occ
#         return loss

#     def get_occ(self, occ_pred, img_metas=None):
#         """
#         Args:
#             occ_pred: (B, Dx, Dy, Dz, C)
#             img_metas:

#         Returns:
#             List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         """
#         occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
#         occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
#         occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
#         return list(occ_res)


# @HEADS.register_module()
# class BEVOCCHead2D(BaseModule):
#     def __init__(self,
#                  in_dim=256,
#                  out_dim=256,
#                  Dz=16,
#                  use_mask=True,
#                  num_classes=18,
#                  use_predicter=True,
#                  class_balance=False,
#                  loss_occ=None,
#                  ):
#         super(BEVOCCHead2D, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.Dz = Dz
#         out_channels = out_dim if use_predicter else num_classes * Dz
#         self.final_conv = ConvModule(
#             self.in_dim,
#             out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=True,
#             conv_cfg=dict(type='Conv2d')
#         )
#         self.use_predicter = use_predicter
#         if use_predicter:
#             self.predicter = nn.Sequential(
#                 nn.Linear(self.out_dim, self.out_dim * 2),
#                 nn.Softplus(),
#                 nn.Linear(self.out_dim * 2, num_classes * Dz),
#             )

#         self.use_mask = use_mask
#         self.num_classes = num_classes

#         self.class_balance = class_balance
#         if self.class_balance:
#             class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
#             self.cls_weights = class_weights
#             loss_occ['class_weight'] = class_weights        # ce loss
#         self.loss_occ = build_loss(loss_occ)

#     def forward(self, img_feats):
#         """
#         Args:
#             img_feats: (B, C, Dy, Dx)

#         Returns:

#         """
#         # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
#         occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
#         bs, Dx, Dy = occ_pred.shape[:3]
#         if self.use_predicter:
#             # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
#             occ_pred = self.predicter(occ_pred)
#             occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

#         return occ_pred

#     def loss(self, occ_pred, voxel_semantics, mask_camera):
#         """
#         Args:
#             occ_pred: (B, Dx, Dy, Dz, n_cls)
#             voxel_semantics: (B, Dx, Dy, Dz)
#             mask_camera: (B, Dx, Dy, Dz)
#         Returns:

#         """
#         loss = dict()
#         voxel_semantics = voxel_semantics.long()
#         if self.use_mask:
#             mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
#             # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
#             voxel_semantics = voxel_semantics.reshape(-1)
#             # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
#             preds = occ_pred.reshape(-1, self.num_classes)
#             # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
#             mask_camera = mask_camera.reshape(-1)

#             if self.class_balance:
#                 valid_voxels = voxel_semantics[mask_camera.bool()]
#                 num_total_samples = 0
#                 for i in range(self.num_classes):
#                     num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
#             else:
#                 num_total_samples = mask_camera.sum()

#             loss_occ = self.loss_occ(
#                 preds,      # (B*Dx*Dy*Dz, n_cls)
#                 voxel_semantics,    # (B*Dx*Dy*Dz, )
#                 mask_camera,        # (B*Dx*Dy*Dz, )
#                 avg_factor=num_total_samples
#             )
#             loss['loss_occ'] = loss_occ
#         else:
#             voxel_semantics = voxel_semantics.reshape(-1)
#             preds = occ_pred.reshape(-1, self.num_classes)

#             if self.class_balance:
#                 num_total_samples = 0
#                 for i in range(self.num_classes):
#                     num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
#             else:
#                 num_total_samples = len(voxel_semantics)

#             loss_occ = self.loss_occ(
#                 preds,
#                 voxel_semantics,
#                 avg_factor=num_total_samples
#             )

#             loss['loss_occ'] = loss_occ
#         return loss

#     def get_occ(self, occ_pred, img_metas=None):
#         """
#         Args:
#             occ_pred: (B, Dx, Dy, Dz, C)
#             img_metas:

#         Returns:
#             List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         """
#         occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
#         occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
#         occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
#         return list(occ_res)


# @HEADS.register_module()
# class BEVOCCHead2D_V2(BaseModule):      # Use stronger loss setting
#     def __init__(self,
#                  in_dim=256,
#                  out_dim=256,
#                  Dz=16,
#                  use_mask=True,
#                  num_classes=18,
#                  use_predicter=True,
#                  class_balance=False,
#                  loss_occ=None,
#                  ):
#         super(BEVOCCHead2D_V2, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.Dz = Dz

#         # voxel-level prediction
#         self.occ_convs = nn.ModuleList()
#         self.final_conv = ConvModule(
#             in_dim,
#             self.out_dim,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=True,
#             conv_cfg=dict(type='Conv2d')
#         )
#         self.use_predicter = use_predicter
#         if use_predicter:
#             self.predicter = nn.Sequential(
#                 nn.Linear(self.out_dim, self.out_dim * 2),
#                 nn.Softplus(),
#                 nn.Linear(self.out_dim * 2, num_classes * Dz),
#             )

#         self.use_mask = use_mask
#         self.num_classes = num_classes

#         self.class_balance = class_balance
#         if self.class_balance:
#             class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
#             self.cls_weights = class_weights
#         self.loss_occ = build_loss(loss_occ)

#     def forward(self, img_feats):
#         """
#         Args:
#             img_feats: (B, C, Dy=200, Dx=200)
#             img_feats: [(B, C, 100, 100), (B, C, 50, 50), (B, C, 25, 25)]   if ms
#         Returns:

#         """
#         # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
#         occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
#         bs, Dx, Dy = occ_pred.shape[:3]
#         if self.use_predicter:
#             # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
#             occ_pred = self.predicter(occ_pred)
#             occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

#         return occ_pred

#     def loss(self, occ_pred, voxel_semantics, mask_camera):
#         """
#         Args:
#             occ_pred: (B, Dx, Dy, Dz, n_cls)
#             voxel_semantics: (B, Dx, Dy, Dz)
#             mask_camera: (B, Dx, Dy, Dz)
#         Returns:

#         """
#         loss = dict()
#         voxel_semantics = voxel_semantics.long()    # (B, Dx, Dy, Dz)
#         preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()    # (B, n_cls, Dx, Dy, Dz)
#         loss_occ = self.loss_occ(
#             preds,
#             voxel_semantics,
#             weight=self.cls_weights.to(preds),
#         ) * 100.0
#         loss['loss_occ'] = loss_occ
#         loss['loss_voxel_sem_scal'] = sem_scal_loss(preds, voxel_semantics)
#         loss['loss_voxel_geo_scal'] = geo_scal_loss(preds, voxel_semantics, non_empty_idx=17)
#         loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)

#         return loss

#     def get_occ(self, occ_pred, img_metas=None):
#         """
#         Args:
#             occ_pred: (B, Dx, Dy, Dz, C)
#             img_metas:

#         Returns:
#             List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         """
#         occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
#         occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
#         occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
#         return list(occ_res)

#     def get_occ_gpu(self, occ_pred, img_metas=None):
#         """
#         Args:
#             occ_pred: (B, Dx, Dy, Dz, C)
#             img_metas:

#         Returns:
#             List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         """
#         occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
#         occ_res = occ_score.argmax(-1).int()      # (B, Dx, Dy, Dz)
#         return list(occ_res)