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
from newscenes_devkit.data_classes import Box as NewScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class NewScenesDataset_MTL(Custom3DDataset):
    r"""NewScenes Dataset.

    This class serves as the API for experiments on the NewScenes Dataset.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """

    NameMapping = {
        "suv":"car",
        "van":"car",
        "truck":"large_vehicle",
        "rider":"rider",
        # 'cyclist':'rider',#----原来翻译成了cyclist这里兼容一下防止漏掉
        "pedestrian":"pedestrian",
        "car":"car",
        "tricyclist":"car",
        "light_truck":"large_vehicle",
        "bus":"large_vehicle",
        "engineering_vehicle":"large_vehicle",
        "handcart":"car",
        "trailer":"large_vehicle",
        } #---与category_to_detection_name保持一致
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',

    }
    
#-------------最终使用标签，这里按此顺序分配label，和配置文件一致-------------
    CLASSES = ('car', 'pedestrian', 'rider', 'large_vehicle')
    
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_newsc_config_final', #-----修改评估的配置文件
                 use_valid_flag=False,
                 #------加入OCC需要的字段-----
                 occ_size=[240, 160, 16], #--加入occ_size
                 pc_range = [-60.0, -60.0, -3.0, 60.0, 60.0, 5.0],#-加入pc_range
                 use_semantic = True,#--使用语义
                 occ_class_names = None,#--OCC的类别

                 ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from newscenes_devkit.eval.detection.config import config_factory #--
        self.eval_detection_configs = config_factory(self.eval_version) #--这里可以根据需求进行修改eval文件
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=True,
                use_map=False,
                use_external=False,
            )
        ## occ_input
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.occ_class_names = occ_class_names

#----------------------这里use_semantic保持兼容性---------------
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



    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            mask = np.full_like(info['valid_flag'], True)
            gt_names = set(info['gt_names'][mask])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids
#----------这里已经按照时间戳顺序排序----------
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos
#----------------------这里根据需求进行修改---------------
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
    #---------------根据valid_flag过滤目标，加入速度，生成LiDARInstance3DBoxes的gt_bboxes_3d-------
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # -------是否根据visibility过滤-----
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = np.full_like(info['valid_flag'], True)
        #---------------------------
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))  #----按顺序排列的标签---
            else:
                gt_labels_3d.append(-1)  #---不在标签内的标签为-1--
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity: #----np.nan的速度全部赋值0----------
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]   
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1) #----bbox和速度拼在一起

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)转换到底边中点，在lidar/ego下-----
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, #--LiDARInstance3DBoxes
            gt_labels_3d=gt_labels_3d, #--array([class_id])
            gt_names=gt_names_3d) #--array([class_name])
        return anns_results
#-----------转成newsc评估的格式，生成ego/lidar下的预测结果，并根据newsc_detection_config.json过滤-------------
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        #----------循环每一帧结果-------------------
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_newsc_box(det,mapped_class_names,self.eval_detection_configs)  #----将LiDARInstance3DBoxes转成newscbox并过滤一些范围外的----
            sample_token = self.data_infos[sample_id]['token']

        #--------------没有设置动静属性标签--------------
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]   #----cat2id----
            #---------这里是最终的newsc格式---------
                nusc_anno = dict(
                    sample_token=sample_token, #---token
                    translation=box.center.tolist(), #---lidar/ego坐标系中心点
                    size=box.wlh.tolist(), #---wlh
                    rotation=box.orientation.elements.tolist(), #----四元数
                    velocity=box.velocity[:2].tolist(), #----vxvy
                    detection_name=name, #----name
                    detection_score=box.score, #---置信度
                                    )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        } #----------每一帧有200多个最初的检测结果，置信度由高到低排---

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_newsc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
    #--------------调用newsc的评估工具----------
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in newScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from newscenes_devkit.newscenes import NewScenes
        from newscenes_devkit.eval.detection.evaluate import NewScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        newsc = NewScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'val_mini',
            'v1.0-trainval': 'val',
        }
        newsc_eval = NewScenesEval(
            newsc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True)
        newsc_eval.main(render_curves=False)

        #-----测试的指标在metrics_summary中------------
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NewScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NOS'.format(metric_prefix)] = metrics['NOS']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
    
    #-----------将结果整理成标准的格式以评估------------
    #-----------这里加入OCC后可能有所变化-------------
    #-----------有可能在这里进行过滤黑天和雨天-----------
    #TODO
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        #-----{'bbox_results': bbox_results, 'occ_results': occ_results}-----
        #----判断哪个task有results-----
        assert isinstance(results, dict), 'results must be a dict'
        if len(results['bbox_results']) != 0:
            assert len(results['bbox_results']) == len(self), (
                'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results['bbox_results']), len(self)))
        if len(results['occ_results']) != 0:
            print('len occ:',len(results['occ_results']))
        #     assert len(results['occ_results']) == len(self), (
        #         'The length of results is not equal to the dataset len: {} != {}'.
        #         format(len(results['bbox_results']), len(self))) 

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        #-------这里是第二种格式--------------------
        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on newScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        
        #---------生成最终结果用于评估,3dOD存了json结果文件---------
        #---------{'3dod':{'pts':xxx/xxx.json},'occ':[N个array]}-----
        final_results = {'3dod':{},'occ':{}}
        if len(results['bbox_results']) != 0:

            if not ('pts_bbox' in results['bbox_results'][0] or 'img_bbox' in results['bbox_results'][0]):
                result_files = self._format_bbox(results['bbox_results'], jsonfile_prefix)
            else:
                # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
                result_files = dict()
                for name in results['bbox_results'][0]:
                    print(f'\nFormating bboxes of {name}') #----'pts_bbox'
                    results_ = [out[name] for out in results['bbox_results']]  #-----[dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)]
                    tmp_file_ = osp.join(jsonfile_prefix, name)
                    result_files.update(
                        {name: self._format_bbox(results_, tmp_file_)}) #---这里调用生成nusc_submissions的json文件
            final_results['3dod'] = result_files
        #-----生成occ结果-----
        if len(results['occ_results']) != 0:
            final_results['occ'] = results['occ_results']
        
        
        return final_results, tmp_dir

#-------------这里加入OCC后要改-----------------
#TODO
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in newScenes protocol.

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
        #--------------生成最终的json文件进行评估-----------------
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        od_results_dict = dict()
        occ_results_dict = dict()
        #----------------评估3d OD---------------------
        if len(result_files['3dod']) != 0:
            print('------------Exit 3D OD Task, Now Evaluate 3d OD!--------------------')

            if isinstance(result_files['3dod'], dict):
                
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files['3dod'][name])
                od_results_dict.update(ret_dict)
            elif isinstance(result_files['3dod'], str):
                od_results_dict = self._evaluate_single(result_files['3dod'])

            if tmp_dir is not None:
                tmp_dir.cleanup()

        #----这里是OCC的评估，这里的result_files['occ']是一个list，每个元素是一个array
        if len(result_files['occ']) != 0:
            print('----------------------Exit OCC Task, Now Evaluate OCC!--------------------')
            
            if self.use_semantic:
                class_names = {0: 'IoU'}
                class_num = len(self.occ_class_names) + 1
                for i, name in enumerate(self.occ_class_names):
                    class_names[i + 1] = self.occ_class_names[i]
                
                results = np.stack(result_files['occ'], axis=0).mean(0) # 平均结果
                # results = np.vstack((results[-1], results[:-1])) # 调整列表位置确保首位是计算非空SSC指标 

                mean_ious = []
                
                for i in range(class_num):
                    tp = results[i, 0]  #   True Positive
                    p = results[i, 1]   #   Predicted
                    g = results[i, 2]   #   Ground Truth
                    union = p + g - tp
                    mean_ious.append(tp / union)
                
                for i in range(class_num):
                    occ_results_dict[class_names[i]] = mean_ious[i]
                occ_results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])
        if logger is not None:
            logger.info(f'Results:\n{od_results_dict}')
            logger.info(f'Results:\n{occ_results_dict}')

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        results_dict = {**od_results_dict, **occ_results_dict}
        return results_dict
    



#-------------未改---------------------------------
    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=6,
                use_dim=[0,1,2,3,5],
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=2,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)
#-----------------------------------------------------------------------
#-----------------这里还没修改-------------------------------------------------
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
#----------------------------------------------------------------------------
#---------------------------将mmdet3d预测结果转换成newscenes的格式---------------------
def output_to_newsc_box(detection,classes,eval_configs):
    """Convert the output to the box class in the newScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
    Returns:
        list[:obj:`NewScenesBox`]: List of standard NewScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy() #---BaseInstance3DBoxes默认底边中点转到中心点
    box_dims = box3d.dims.numpy() #---[xsize, ysize, zsize],newsc是wlh
    box_yaw = box3d.yaw.numpy() #--朝向角
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    #----这里再转回到newsc初始的时候，因为读进来的时候转换过一次以适配LiDARInstance3DBoxes
    #-----这里有小于pi的值，-4.多,是否需要用limit_period
    box_yaw = -box_yaw - np.pi / 2 

    box_list = []
    for i in range(len(box3d)):
        #--yaw转换成四元数，需要注意的这个范围超了不会影响
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i]) 
        velocity = (*box3d.tensor[i, 7:9], 0.0) #--传入tensor不影响
        box = NewScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],  #----------这个label要和classes中的顺序对应
            score=scores[i],
            velocity=velocity)
        cls_range_map = eval_configs.class_range #---不同类别的范围（可以根据标注的每类最远距离来确定）
        det_range = cls_range_map[classes[box.label]] #---过滤掉ego坐标系下标注范围以外的目标,与评估一致---
        if abs(box.center[0])> det_range[0] or abs(box.center[1])> det_range[1]:
            continue

        # radius = np.linalg.norm(box.center[:2], 2)
        # if radius > det_range:
        #     continue
        
        box_list.append(box)
    return box_list  #-----返回NewScenesBox实例，显示的是字符串所有信息


