
import mmcv
import numpy as np
import os
from collections import OrderedDict

from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img

from newscenes_devkit.newscenes import NewScenes
from projects.mmdet3d_plugin.datasets.newscenes_dataset import NewScenesDataset
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Written by [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


def create_newscenes_infos(root_path,
                          out_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of newscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 3
    """
    print(version, root_path)

   
    newsc = NewScenes(version=version, dataroot=root_path, verbose=True)

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = newsc.scene_split['train']
        val_scenes = newsc.scene_split['val']
    elif version == 'v1.0-test':
        train_scenes = newsc.scene_split['test']
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = newsc.scene_split['train_mini']
        val_scenes = newsc.scene_split['val_mini']
    else:
        raise ValueError('unknown')

# -----------if existing scenes.-------------------------
    all_scenes = sorted(next(os.walk(newsc.dataroot))[1])
    if all(scene in all_scenes for scene in train_scenes) and \
    val_scenes != [] and all(scene in all_scenes for scene in val_scenes):
        print('Train and Val scenes exist.')
    elif all(scene in all_scenes for scene in train_scenes) and val_scenes == []:
        print('Test scenes exist.')
    else:
        raise ValueError('Some scenes do not exist.')
#--------------------------------------------
    
    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        newsc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    
    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)



def _get_can_bus_info(lidar_pose_record, canbus_record):
    rotation, translation = RT_transform_to_quaternion(lidar_pose_record['pose'])
    acc_xyz = canbus_record['acc_xyz']
    gyro_xyz = canbus_record['gyro_xyz']
    velocity_ego = canbus_record['velocity_ego']

    can_bus = translation + rotation + acc_xyz + gyro_xyz + velocity_ego + [0., 0.]

    return np.array(can_bus)


def _fill_trainval_infos(newsc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        newsc (:obj:`NewScenes`): Dataset class in the newScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_newsc_infos = []
    val_newsc_infos = []

    
    for sample in mmcv.track_iter_progress(newsc.sample):
        lidar_token = sample['token']    
        sync_record = newsc.get('sample_data', sample['token'])
        
        lidar_pose_record = newsc.get('ego_pose', sync_record['ego_pose']['lidar_top_compensation'],sample['scene_token'])
        canbus_record = newsc.get('imu_data', sync_record['ego_pose']['lidar_top_compensation'],sample['scene_token'])
                 
  
        lidar_path = osp.join(newsc.dataroot, sync_record['lidar']['lidar_top_compensation'])
        boxes = newsc.get_annotation_box(lidar_token)

        mmcv.check_file_exist(lidar_path)

    
        can_bus = _get_can_bus_info(lidar_pose_record, canbus_record)
        
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': sample['frame_idx'],  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'radars':dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': can_bus[:3],
            'ego2global_rotation': can_bus[3:7],
            'timestamp': sample['timestamp'],
        }


        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix



        camera_types = ['camera_front', 
                        'camera_left_front',
                        'camera_right_front', 
                        'camera_back', 
                        'camera_left_back', 
                        'camera_right_back']
        for cam in camera_types:
            #TODO 根据scene_token取对应的相机内参和畸变参数
            cam_intrinsic = newsc.get('sensor_calibration',info['scene_token'])['calib'][cam]['intrinsic']
            cam_distortion = newsc.get('sensor_calibration',info['scene_token'])['calib'][cam]['distortion']

            cam_info = obtain_sensor2top(newsc, sync_record['token'], info['scene_token'], l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            cam_info.update(cam_distortion=cam_distortion)
            info['cams'].update({cam: cam_info})


        radar_types = ['radar_front', 
                       'radar_left_front', 
                       'radar_right_front', 
                       'radar_back', 
                       'radar_left_back', 
                       'radar_right_back']

        for radar_name in radar_types:
            radar_token = sync_record['token']  
            sweeps = []
            
            while len(sweeps) < 3:
                if not newsc.get('sample_data', radar_token)['prev'] == '':
                
                    radar_info = obtain_sensor2top(newsc, radar_token, info['scene_token'], l2e_t, l2e_r_mat,
                                                e2g_t, e2g_r_mat, radar_name)
                    sweeps.append(radar_info)
                    radar_token = newsc.get('sample_data', radar_token)['prev'] #---token赋值成前一帧sync
                else:
                    radar_info = obtain_sensor2top(newsc, radar_token, info['scene_token'], l2e_t, l2e_r_mat,
                                                e2g_t, e2g_r_mat, radar_name)
                    sweeps.append(radar_info)
            
            info['radars'].update({radar_name: sweeps})


        lidar_sweeps = []
        lidar_token = sync_record['token']  
        while len(lidar_sweeps) < max_sweeps:
            if not newsc.get('sample_data', lidar_token)['prev'] == '':
                sweep = obtain_sensor2top(newsc, newsc.get('sample_data', lidar_token)['prev'], info['scene_token'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                lidar_sweeps.append(sweep)
                lidar_token = newsc.get('sample_data', lidar_token)['prev']
            else:
                break
        info['sweeps'] = lidar_sweeps
        
        
        if not test:

            
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)

            velocity = np.array([b.velocity[:2] for b in boxes]).reshape(-1, 2)

   
            names = [b.name for b in boxes]
            
            for i in range(len(names)):
                if names[i] in NewScenesDataset.NameMapping:
                    names[i] = NewScenesDataset.NameMapping[names[i]]
                else:
                    print(f'不存在此类{names[i]}')
                    return
            names = np.array(names)
            # we need to convert rot to SECOND format.
            
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                boxes), f'{len(gt_boxes)}, {len(boxes)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)



            visibility = np.array([b.visibility
                             for b in boxes],dtype=bool)
            info['visibility'] = visibility
            #---------------------------------------------------------------------------
            info['num_lidar_pts'] = np.full(len(gt_boxes), -1, dtype=int)
            info['num_radar_pts'] = np.full(len(gt_boxes), -1, dtype=int)
            info['valid_flag'] = visibility 
#------------------------------------------------------------------------------
        if sample['scene_token'] in train_scenes:
            train_newsc_infos.append(info)
        if sample['scene_token'] in val_scenes:
            val_newsc_infos.append(info)  
        #TODO 有测试集时的处理

    return train_newsc_infos, val_newsc_infos




def obtain_sensor2top(newsc,
                      sync_token,
                      scene_token, 
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        newsc (class): Dataset class in the newScenes dataset.
        sync_token (str): sync token corresponding to the
            sync file.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sync_record = newsc.get('sample_data', sync_token)
    
    if sensor_type == 'lidar':
        data_path = osp.join(newsc.dataroot, sync_record['lidar']['lidar_top_compensation'])
        cs_record = {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]}
        ego_pose_token = sync_record['ego_pose']['lidar_top_compensation']

        sensor_timestamp = sync_token 
        
    
        
    elif sensor_type[:3] == 'cam':
        data_path = osp.join(newsc.dataroot, sync_record['cameras'][sensor_type])
        RT_matrix = newsc.get('sensor_calibration',scene_token)['calib'][sensor_type]['camera2ego']
        rotation_cam, translation_cam = RT_transform_to_quaternion(RT_matrix) 
        cs_record = {'translation': translation_cam, 
                     'rotation': rotation_cam}
        ego_pose_token = sync_record['ego_pose']['lidar_top_compensation']

        sensor_timestamp = sync_token  

    
    elif sensor_type[:3] == 'rad':
        data_path = osp.join(newsc.dataroot, sync_record['radars'][sensor_type])
        RT_matrix = newsc.get('sensor_calibration',scene_token)['calib'][sensor_type]['radar2ego']
        rotation_rad, translation_rad = RT_transform_to_quaternion(RT_matrix)
        cs_record = {'translation': translation_rad, 
                     'rotation': rotation_rad}
        ego_pose_token = sync_record['ego_pose'][sensor_type]
        sensor_timestamp = sync_record['radars'][sensor_type].split('/')[-1][:-4]

    else:
        raise ValueError('unknown sensor type')

    

    pose_matrix = newsc.get('ego_pose', ego_pose_token,scene_token)['pose']
    canbus_record = newsc.get('imu_data', ego_pose_token,scene_token) 
    rotation_pose, translation_pose = RT_transform_to_quaternion(pose_matrix)

    pose_record = {'translation': translation_pose, 
                     'rotation': rotation_pose}
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sync_token,  
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'ego_velocity':canbus_record['velocity_ego'], 
        'timestamp': sensor_timestamp 
    }
    
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # 这里计算过程是(e2l*g2e*e2g*s2e).T.T
    sweep['sensor2lidar_translation'] = T
    return sweep


def RT_transform_to_quaternion(RT_matrix: List[float]) -> List[float]:
    """Convert the RT matrix to quaternion.
    """
    transformation_matrix = np.array(RT_matrix)
    if transformation_matrix.shape != (4, 4):
        transformation_matrix = transformation_matrix.reshape(4, 4)

    quaternion = Quaternion(matrix=transformation_matrix[:3, :3], atol=1e-4) 
    rotation = quaternion.elements.astype(float).tolist()     
    translation = transformation_matrix[:3, 3].astype(float).tolist()       
    return rotation, translation



def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None




if __name__ == '__main__':
    root_path = 'data/NewScenes_Final'
    out_path = 'data/NewScenes_Final'
    info_prefix = 'newscenes-final'
    version = 'v1.0-trainval'
    max_sweeps = 2  



    create_newscenes_infos(root_path,
                          out_path,
                          info_prefix,
                          version,
                          max_sweeps)
