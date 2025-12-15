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
        nonempty_target = ssc_target != 0
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
        mask = ssc_target != 255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :]
            # alternative format: p = pred[:, :, :, :, i]

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
        nonempty_target = ssc_target != 0
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
        mask = ssc_target != 255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :]
            # alternative format: p = pred[:, :, :, :, i]

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
        nonempty_target = ssc_target != 0  
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
        mask = ssc_target != 255    
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] ## surroundocc style
            # p = pred[:, :, :, :, i] ## bevformer_occ style

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
        # loss['loss_occ'] = total_loss   


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
        nonempty_target = ssc_target != 0  
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
        mask = ssc_target != 255    
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] 
            # p = pred[:, :, :, :, i] 

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
        # loss['loss_occ'] = total_loss   


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
        nonempty_target = ssc_target != 0  
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
        mask = ssc_target != 255    
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :] 
            # p = pred[:, :, :, :, i] 

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

