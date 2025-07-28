import os
from multiprocessing import Pool

import mmcv
import numpy as np
from newscenes_devkit.data_classes import LidarPointCloud
from newscenes_devkit.geometry_utils import view_points
from pyquaternion import Quaternion
import copy
import cv2
from tqdm import tqdm
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    lidar2ego_translation,
    lidar2ego_rotation,
    ego2global_translation,
    ego2global_rotation,
    sensor2ego_translation, 
    sensor2ego_rotation,
    cam_ego2global_translation,
    cam_ego2global_rotation,
    cam_intrinsic,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar2ego_rotation).rotation_matrix)
    pc.translate(np.array(lidar2ego_translation))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(ego2global_rotation).rotation_matrix)
    pc.translate(np.array(ego2global_translation))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego2global_translation))
    pc.rotate(Quaternion(cam_ego2global_rotation).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(sensor2ego_translation))
    pc.rotate(Quaternion(sensor2ego_rotation).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_intrinsic),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


data_root = './data/NewScenes_Final'
info_path_train = './data/NewScenes_Final/newscenes-final_infos_temporal_train.pkl'
info_path_val = './data/NewScenes_Final/newscenes-final_infos_temporal_val.pkl'

# data3d_nusc = NuscMVDetData()

# lidar_key = 'LIDAR_TOP'
cam_keys = [
    'camera_left_front', 'camera_front', 'camera_right_front', 'camera_right_back',
    'camera_back', 'camera_left_back'
]


def worker(info):
    lidar_path = info['lidar_path']
    points = np.fromfile(lidar_path,
                         dtype=np.float32,
                         count=-1).reshape(-1, 6)[..., :4]
    
    lidar2ego_translation = info['lidar2ego_translation']
    lidar2ego_rotation = info['lidar2ego_rotation']
    ego2global_translation = info['ego2global_translation']
    ego2global_rotation = info['ego2global_rotation']
    for i, cam_key in enumerate(cam_keys):
        sensor2ego_translation = info['cams'][cam_key]['sensor2ego_translation']
        sensor2ego_rotation = info['cams'][cam_key]['sensor2ego_rotation']
        cam_ego2global_translation = info['cams'][cam_key]['ego2global_translation']
        cam_ego2global_rotation = info['cams'][cam_key]['ego2global_rotation']
        cam_intrinsic = info['cams'][cam_key]['cam_intrinsic']
        current_img = mmcv.imread(
            os.path.join(info['cams'][cam_key]['data_path']))
        img = cv2.undistort(current_img, np.array(info['cams'][cam_key]['cam_intrinsic'])[:3,:3],\
                                                      np.array(info['cams'][cam_key]['cam_distortion']), None, np.array(info['cams'][cam_key]['cam_intrinsic'])[:3,:3])
        pts_img, depth = map_pointcloud_to_image(
            points.copy(), img, 
            copy.deepcopy(lidar2ego_translation), 
            copy.deepcopy(lidar2ego_rotation), 
            copy.deepcopy(ego2global_translation),
            copy.deepcopy(ego2global_rotation),
            copy.deepcopy(sensor2ego_translation), 
            copy.deepcopy(sensor2ego_rotation), 
            copy.deepcopy(cam_ego2global_translation), 
            copy.deepcopy(cam_ego2global_rotation),
            copy.deepcopy(cam_intrinsic))
        
        # file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
        # np.concatenate([pts_img[:2, :].T, depth[:, None]],
        #                axis=1).astype(np.float32).flatten().tofile(
        #                    os.path.join('./data', 'depth_gt_newscenes',
        #                                 f'{file_name}.bin'))
        current_dir = os.path.dirname(info['cams'][cam_key]['data_path']).replace("cameras","depth_gt")
        mmcv.mkdir_or_exist(current_dir)
        np.concatenate([pts_img[:2, :].T, depth[:, None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           info['cams'][cam_key]['data_path'].replace("cameras","depth_gt") + ".bin")
        

if __name__ == '__main__':
    ## debug_test
    # infos = mmcv.load(info_path_train)['infos']
    # for info in tqdm(infos):
    #     worker(info)
    ### 
    po = Pool(12)

    infos = mmcv.load(info_path_train)['infos']
    for info in tqdm(infos):
        po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()
    
    # po2 = Pool(12)
    # infos = mmcv.load(info_path_val)['infos']
    # for info in tqdm(infos):
    #     po2.apply_async(func=worker, args=(info, ))
    # po2.close()
    # po2.join()
