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
from projects.mmdet3d_plugin.datasets import NewScenesDataset_MTL
from newscenes_devkit.eval.common.utils import quaternion_yaw, Quaternion
from newscenes_devkit.geometry_utils import transform_matrix
@DATASETS.register_module()
class CustomNewScenesDataset_MTL(NewScenesDataset_MTL):

    def __init__(self, queue_length=4, bev_size=(160, 240), overlap_test=False, *args, **kwargs):
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
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict) 
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)


    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        # multi-frame points
        if 'points' in queue[0].keys():
            points_list = [each['points'] for each in queue]
        
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None

        ego2global_transform_lst = []  # add ego pose list

        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            ego2global_transform_lst.append(metas_map[i]['ego2global_transformation'])
            # Handle missing previous frames and scene boundaries; store relative pos/angle
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
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
            # Merge queue images and img_metas into the last entry (current frame)
            queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)

        # add points if present
        if 'points' in queue[0].keys():
            queue[-1]['points'] = points_list
        # add ego2global transformation list to the last meta
        metas_map[len(queue)-1]["ego2global_transform_lst"] = ego2global_transform_lst

        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]  # keep only last frame (current);
        return queue
    # during testing only single-frame input is used
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
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],

            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],

            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=int(info['timestamp']) / 1e6,

            # occ related fields
            occ_path = info['occ_path'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
        )
        # process pose related transforms; lidar and ego share same coords in newscenes
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar
        
        # process ego2global and lidar2ego transformation
        ego2global_transformation = Quaternion(input_dict['ego2global_rotation']).transformation_matrix
        ego2global_transformation[:3, 3] = input_dict['ego2global_translation']

        lidar2ego_transformation = Quaternion(input_dict["lidar2ego_rotation"]).transformation_matrix
        lidar2ego_transformation[:3, 3] = input_dict['lidar2ego_translation']

        input_dict.update({
            "ego2global_transformation": np.array(ego2global_transformation, dtype=np.float32),
            "lidar2ego_transformation": np.array(lidar2ego_transformation, dtype=np.float32),
        })

        #------------------------------------------------------------


        # add radar data
        if self.modality['use_radar']: 
            input_dict['radars'] = info['radars']
        # add image info: image paths, lidar2cam, lidar2img, camera intrinsics; include distortion coeffs
        # note: front/rear images may be resized (e.g. to half); intrinsics remain unchanged, so lidar2img should be scaled accordingly
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_distortion = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
    # equivalent to the inverse of a 4x4 matrix; extrinsics in the pkl already account for ego-pose timing
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])  # rotation matrix R
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = np.array(cam_info['cam_intrinsic'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)  # effectively K * lidar2cam; inverse of lidar2cam_rt yields homogeneous row [0,0,0,1]
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
        
       
        can_bus = input_dict['can_bus']
        can_bus[:3] = input_dict['ego2global_translation']
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

            data = self.prepare_train_data(idx) 
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

