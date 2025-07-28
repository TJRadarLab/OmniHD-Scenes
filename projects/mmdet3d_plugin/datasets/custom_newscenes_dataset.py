# Copyright (c) OpenMMLab. All rights reserved.

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Written by [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

import mmcv
import numpy as np
import pyquaternion
import tempfile
import random
from newscenes_devkit.data_classes import Box as NewScenesBox
from os import path as osp
import copy
import torch
from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.datasets import NewScenesDataset
from newscenes_devkit.eval.common.utils import quaternion_yaw, Quaternion

@DATASETS.register_module()
class CustomNewScenesDataset(NewScenesDataset):

    def __init__(self, queue_length=4, bev_size=(140, 180), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
    def prepare_train_data(self, index):#-----19436-----
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))#---[19433, 19434, 19435]---
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)     #-----随机过去三帧中的两帧，队列总共3帧,这里是[19434, 19435,19436]------------
        for i in index_list:
            i = max(0, i) #---这里防止index是0,1,2时报错，也就是整个2万多关键帧的前3个--
            input_dict = self.get_data_info(i) #--生成输入的input_dict-----
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)  #---增加custom3d中img_fields等各个字段--
#---根据pipeline中顺序进行数据读取增强组装等，生成img_metas,gt_bboxes_3d,gt_labels_3d,img四个DC----
            example = self.pipeline(input_dict) 
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)


    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]  #-----取出img中的三个所有图像tensor----
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data  #----取出对应的img_metas ----
            #----这里应该是解决这几种情况，首先队列中的第一个肯定没有之前帧，-----
            #-----------其次，如果有一帧是clip最后一帧，下一帧也是没有之前帧的---
            #---------这里将pos和angle记录为和之前的差，没有前一帧就是0------
            if metas_map[i]['scene_token'] != prev_scene_token:  
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos #-----这里相减----
                metas_map[i]['can_bus'][-1] -= prev_angle #----这里相减----
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        #-------------这里将队列中的img和所有img_metas合并到最后一个样本，也就是当前帧----
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)# -[3,6,3,736,1280]
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]   #----这里是annos按时间排好顺序的-----19434
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=int(info['timestamp']) / 1e6,
        )
        #-------加入radar数据------
        if self.modality['use_radar']: 
            input_dict['radars'] = info['radars']
        #------加入图像信息，包括图像路径，lidar2cam，lidar2img，cam内参,这里要注意加入畸变系数------
        #------内参应用要注意，前视后视读取后已经变成一半尺寸，但是内参没变，lidar2img乘了系数------
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_distortion = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
    #---------等效于直接4*4求逆矩阵，这里的平移旋转外参在pkl文件时已经是考虑了不同时刻egopose---------------
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation']) #这个就是R矩阵
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = np.array(cam_info['cam_intrinsic'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T) #--实际就是K*lidar2cam，这里lidar2cam_rt求逆才是第四行0001---
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                cam_distortion.append(np.array(cam_info['cam_distortion']))

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    cam_distortion=cam_distortion,
                ))

        if not self.test_mode:
            
            annos = self.get_ann_info(index)  
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
 
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = input_dict['ego2global_rotation'] 

        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)  #-----调到此函数----
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data




    