
# ---------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ---------------------------------------------

import numpy as np
import mmcv
from pyquaternion import Quaternion
from mmdet.datasets.builder import PIPELINES
from projects.mmdet3d_plugin.core.points.radar_points import RadarPoints
import cv2
from projects.mmdet3d_plugin.core.vis_tools import project_pts_on_img
import torch
from mmdet3d.core.points import BasePoints, get_points_type

# Load ground-truth depth maps; read at resolution 1080x1920
@PIPELINES.register_module()
class LoadGTDepth(object):
    def __init__(self, 
                 scale,
                 pad=4,
                 scale_factor_frontandback=0.5,
                 depth_dim=(1080,1920),
                 ):
        self.scale_factor_frontandback = scale_factor_frontandback
        self.depth_dim = depth_dim
        self.scale = scale
        self.pad = pad

    def __call__(self, results):
        gt_depths = []
        filename = results['filename'].copy()
        for i, name in enumerate(filename):
            cam_depth = np.fromfile(name.replace("cameras","depth_gt")+".bin",
                dtype=np.float32,
                count=-1).reshape(-1, 3)
        # [u, v, d]
            if name.split('/')[-2] == 'camera_front' or name.split('/')[-2] == 'camera_back':
                cam_depth[:, :2] = cam_depth[:, :2] * self.scale_factor_frontandback
            #------------------
            cam_depth[:, :2] = cam_depth[:, :2] * self.scale
            depth_coords = cam_depth[:, :2].astype(np.int16)
            depth_dim = (int(self.depth_dim[0] * self.scale),int(self.depth_dim[1] * self.scale))
            depth_map = np.zeros(depth_dim)
            valid_mask = ((depth_coords[:, 1] < depth_dim[0])
                        & (depth_coords[:, 0] < depth_dim[1])
                        & (depth_coords[:, 1] >= 0)
                        & (depth_coords[:, 0] >= 0))
            depth_map[depth_coords[valid_mask, 1],
                    depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]
            if self.scale == 0.5:
                depth_map = np.pad(depth_map, ((self.pad//2, self.pad//2), (0, 0)),'constant',constant_values = (0,0))
            gt_depths.append(torch.Tensor(depth_map))
        
        gt_depths = torch.stack(gt_depths)  # shape: (6, 1080, 1920)
        results['img_depth'] = gt_depths
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str  






@PIPELINES.register_module()
class LoadOccupancy_Newscenes(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class. 
    """

    def __init__(self, 
                 use_semantic=True,
                #  num_classes=18,
                 class_names=None,
                 occ_size = [240,160,16],
                 ):
        self.use_semantic = use_semantic
        self.class_names  = class_names
        self.num_classes = len(class_names) + 1 # free_space
        self.occ_size = occ_size

    def gt_to_voxel(self, gt, num_classes, occ_size):
        voxel = np.zeros(occ_size)
        voxel[gt[:, 0].astype(np.int), gt[:, 1].astype(np.int), gt[:, 2].astype(np.int)] = gt[:, 3]

        return voxel
    
    def __call__(self, results):
        occ = np.load(results['occ_path'])["occ_gt"]
        occ = occ.astype(np.float32)    # (42319, 4)
        # occ[...,-1] += 1
        semantics = self.gt_to_voxel(occ, self.num_classes, self.occ_size)  # create empty occ voxel grid (200, 200, 16)

        results['gt_occ'] = semantics  # voxel semantic map in range [0, num_classes]
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    



# Original format: [x, y, z, velocity, power, motion_state, SNR, valid_flag]
# Output format: x, y, z, vx_r_comp, vy_r_comp, power, snr, time_diff, Vr (radial), radar_ID
# We use compensated velocities (vx_r_comp, vy_r_comp) and rotate them into ego/lidar frame
@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=8,
                 use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-72, -56, -3.0, 72, 56, 5.0], 
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range

    # Load radar point clouds from .bin files (8 dims)
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 8]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radars']
        radar_ID = {'radar_front':0, 
                       'radar_left_front':1, 
                       'radar_right_front':2, 
                       'radar_back':3, 
                       'radar_left_back':4, 
                       'radar_right_back':5}
        
        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))
            
            ts = int(sweeps[0]['timestamp']) * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = int(sweep['timestamp']) * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

    # Perform velocity compensation and use compensated velocities; transform to current frame (lidar/ego)
    # Project relative radial velocity and compensated velocity into the current frame
                # velocity compensated by the ego motion in sensor frame
                # extract xyz coordinates and radial velocity vr
                xyz = points_sweep[:, :3]
                vr = points_sweep[:, 3]
                # compute distance r
                r = np.linalg.norm(xyz, axis=1)

                # compute azimuth
                azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])

                # compute elevation
                elevation = np.arcsin(xyz[:, 2] / r)

                # decompose ego velocity into the sensor frame
                V_ego = np.array(sweep['ego_velocity']).reshape(-1,3)

                V_sensor = V_ego @ np.linalg.inv(Quaternion(sweep['sensor2ego_rotation']).rotation_matrix).T 
                V_sensor = np.repeat(V_sensor,points_sweep.shape[0], axis=0)
                # compute the component of the ego sensor velocity along radial direction for compensation
                Vr_compensated = V_sensor[:,0] * np.cos(azimuth) * np.cos(elevation) + \
                    V_sensor[:,1] * np.sin(azimuth) * np.cos(elevation) + \
                        V_sensor[:,2] * np.sin(elevation) + vr
                # compute compensated x-direction velocity Vx_compensated
                Vx_compensated = Vr_compensated * np.cos(elevation) * np.cos(azimuth) 
                # compute compensated y-direction velocity Vy_compensated
                Vy_compensated = Vr_compensated * np.cos(elevation) * np.sin(azimuth)

                velo_comp = np.concatenate((Vx_compensated.reshape(-1, 1), Vy_compensated.reshape(-1, 1)), axis=1)            
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                # rotate velocities into the current-frame lidar coordinates
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]


# Transform points into the current frame
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                points_radarID = np.full((points_sweep.shape[0], 1), radar_ID[key])
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :3], 
                     velo_comp, points_sweep[:, [4,6]],
                     time_diff,Vr_compensated.reshape(-1, 1),points_radarID], axis=1)  # format: x y z vx_r_comp vy_r_comp power snr time_diff Vr radar_ID
                points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)  # ~8000-20000 points
        
        points = points[:, self.use_dim]
        
        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        

        
        radar_mask = points.in_range_3d(self.pc_range)
        clean_radar = points[radar_mask]



        results['points'] = clean_radar  # assign filtered radar points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'



@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_newsc(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        img = []
        lidar2img = []
        cam_intrinsic = []
        scale_factor_frontandback = 0.5
        for i, name in enumerate(filename):
            current_img = mmcv.imread(name, self.color_type)
            # cv2.imwrite(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/saved_image_ori_{i}.jpg', current_img)
            current_img_undistortion = cv2.undistort(current_img, results['cam_intrinsic'][i][:3,:3],\
                                                      results['cam_distortion'][i], None, results['cam_intrinsic'][i][:3,:3])
            # cv2.imwrite(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/saved_image_dist_{i}.jpg', current_img_undistortion)
            if name.split('/')[-2] == 'camera_front' or name.split('/')[-2] == 'camera_back':
                img_final = mmcv.imresize(current_img_undistortion, (int(current_img_undistortion.shape[1] * scale_factor_frontandback), int(current_img_undistortion.shape[0] * scale_factor_frontandback)), return_scale=False)
                # cv2.imwrite(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/saved_image_resize_{i}.jpg', img_final)
                img.append(img_final)
                scale_factor = np.eye(4)
                scale_factor[0, 0] *= scale_factor_frontandback
                scale_factor[1, 1] *= scale_factor_frontandback
                current_lidar2img = scale_factor @ results['lidar2img'][i]
                cam_intrinsic.append(scale_factor @ results['cam_intrinsic'][i])
                lidar2img.append(current_lidar2img)
            else:
                img.append(current_img_undistortion)
                lidar2img.append(results['lidar2img'][i])
                cam_intrinsic.append(results['cam_intrinsic'][i])
        
        img = np.stack(img, axis=-1) #----（1080,1920,3,6）---
        results['lidar2img'] = lidar2img
        results['cam_intrinsic'] = cam_intrinsic
        if self.to_float32:
            img = img.astype(np.float32) 
        results['filename'] = filename 
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])] 
        results['img_shape'] = img.shape  
        results['ori_shape'] = img.shape #----（1080,1920,3,6）
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape #----（1080,1920,3,6）
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False) 

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
    

@PIPELINES.register_module()
class LoadPointsFromFile_reducedbeams(object):

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 
                 reduce_beams_to=32,
                 chosen_beam_id=13,

                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        self.reduce_beams_to = reduce_beams_to
        self.chosen_beam_id = chosen_beam_id
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        points = reduce_LiDAR_beams(points, self.reduce_beams_to, self.chosen_beam_id)
        

        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


def reduce_LiDAR_beams(pts, reduce_beams_to=32, chosen_beam_id=13):
    beam_range = [-25, -19.582, -16.042, -13.565, -11.742, -10.346, -9.244, -8.352, -7.65, -7.15, -6.85, -6.65, -6.5, -6.39, -6.29, -6.19, -6.09, -5.99, -5.89, -5.79, -5.69, -5.59, -5.49, -5.39, -5.29, -5.19, -5.09, -4.99, -4.89, -4.79, -4.69, -4.59, -4.49, -4.39, -4.29, -4.19, -4.09, -3.99, -3.89, -3.79, -3.69, -3.59, -3.49, -3.39, -3.29, -3.19, -3.09, -2.99, -2.89, -2.79, -2.69, -2.59, -2.49, -2.39, -2.29, -2.19, -2.09, -1.99, -1.89, -1.79, -1.69, -1.59, -1.49, -1.39, -1.29, -1.19, -1.09, -0.99, -0.89, -0.79, -0.69, -0.59, -0.49, -0.39, -0.29, -0.19, -0.09, 0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91, 2.01, 2.11, 2.21, 2.31, 2.41, 2.51, 2.61, 2.71, 2.81, 2.91, 3.01, 3.11, 3.21, 3.31, 3.41, 3.51, 3.61, 3.71, 3.81, 3.96, 4.16, 4.41, 4.71, 5.06, 5.46, 5.96, 6.56, 7.41, 9, 11.5, 15]
    beam_range = np.radians(beam_range)
    beam_range = np.sort(beam_range)[::-1]
    
    lidar2ego = np.array([
        [0.999648, 0.019996, -0.017452, 1.26],
        [-0.019999, 0.9998, 0, 0],
        [0.017449, 0.000349, 0.999848, 1.855],
        [0, 0, 0, 1]])
    ego2lidar = np.linalg.inv(lidar2ego)
    pts = transform_points(pts, ego2lidar)
    #print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    beam_range = torch.tensor(beam_range.copy())
    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        liat_16 = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127]
        for id in liat_16:
            beam_mask = (theta < (beam_range[id-1]-0.000873)) * (theta > (beam_range[id]-0.000873))
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        liat_4 = [31, 63, 95, 127]
        for id in liat_4:
            beam_mask = (theta < (beam_range[id-1]-0.000873)) * (theta > (beam_range[id]-0.000873))
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 32:
        liat_32 = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
        for id in liat_32:
            beam_mask = (theta < (beam_range[id-1]-0.000873)) * (theta > (beam_range[id]-0.000873))
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 64:
        list_64 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127]
        for id in list_64:
            beam_mask = (theta < (beam_range[id-1]-0.000873)) * (theta > (beam_range[id]-0.000873))
            mask = mask + beam_mask
        mask = mask.bool()
    else:
        lim_range = list(range(30, 90, 1))
        for id in lim_range:
            beam_mask = (theta < (beam_range[id-1]-0.000873)) * (theta > (beam_range[id]-0.000873))
            mask = mask + beam_mask
        mask = mask.bool()
    points = pts[mask]
    points = transform_points(points.numpy(), lidar2ego)
    return points


def transform_points(points, RT_matrix): 
    points_3d = points[:, :3]
    transformed_xyz = np.matmul(points_3d, RT_matrix[:3,:3].T)
    transformed_xyz += RT_matrix[:3, 3]
    points_transformed = np.hstack((transformed_xyz, points[:, 3:]))
    return points_transformed
