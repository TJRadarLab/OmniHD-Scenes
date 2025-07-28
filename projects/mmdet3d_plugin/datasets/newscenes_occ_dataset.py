# Copyright (c) OpenMMLab. All rights reserved.

# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------

import mmcv
import numpy as np
import pyquaternion
import tempfile
from newscenes_devkit.data_classes import Box as NewScenesBox
### 需要专门设置一个git项目用于保存newscenes_devkit项目，偏于各项目迁移
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from projects.mmdet3d_plugin.datasets import NewScenesDataset


@DATASETS.register_module()
class NewScenesOccDataset(NewScenesDataset):

    def __init__(self,
                occ_size=[240, 160, 16],
                pc_range = [-60.0, -60.0, -5.0, 60.0, 60.0, 3.0],
                use_semantic = True,
                classes = None,
                overlap_test=False,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        ## occ_input
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self.overlap_test = overlap_test

#----------------------这里根据需求进行修改---------------
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if not self.use_semantic:
            if self.filter_empty_gt and \
                    (example is None or
                        ~(example['gt_labels_3d']._data != -1).any()):
                return None
        return example
    
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
        
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=int(info['timestamp']) / 1e6,
            ## occ_input
            occ_path = info['occ_path'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
        )

        #-------加入radar数据------
        if self.modality['use_radar']: 
            input_dict['radars'] = info['radars']
        #------加入图像数据，这里要注意加入畸变系数------------
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

        if not self.test_mode: #-----------这里控制是不是测试集无标签----
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    # ### 实验-小批量数据
    # def __len__(self):
    #     """Return the length of data infos.

    #     Returns:
    #         int: Length of data infos.
    #     """
    #     return 100
    
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {0: 'IoU'}
            class_num = len(self.class_names) + 1
            for i, name in enumerate(self.class_names):
                class_names[i + 1] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0) # 平均结果
            # results = np.vstack((results[-1], results[:-1])) # 调整列表位置确保首位是计算非空SSC指标 

            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0]  #   True Positive
                p = results[i, 1]   #   Predicted
                g = results[i, 2]   #   Ground Truth
                union = p + g - tp
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])
        if logger is not None:
            logger.info(f'Results:\n{results_dict}')
        return results_dict
