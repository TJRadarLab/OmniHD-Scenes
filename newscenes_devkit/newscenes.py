import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from newscenes_devkit.data_classes import LidarPointCloud, RadarPointCloud, Box
from newscenes_devkit.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix, transform_points

from collections import OrderedDict
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Modified by [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("NewScenes dev-kit only supports Python version 3.")


class NewScenes:
    """
    Database class for NewScenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/newscenes-mini',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        #-----json列表----
        self.table_names = ['sample', 'sample_data', 'annotations','ego_pose','imu_data','sensor_calibration','meta']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NewScenes tables for version {} Modified by [TONGJI] [Lianqing Zheng]...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.annotations = self.__load_table__('annotations')

        self.ego_pose = self.__load_table__('ego_pose')
        self.imu_data = self.__load_table__('imu_data')
        self.scene_split = self.__load_table__('scene_split')
        self.sensor_calibration = self.__load_table__('sensor_calibration')
        self.meta = self.__load_table__('meta')


        if verbose:
            for table in self.table_names:

                print("{} {},".format(len(getattr(self, table)), table))
                if table == 'sample_data':
                    print("{} {},".format(len(getattr(self, table))*13, 'sync_sensor_frames'))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

       


    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.

        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()
            if table in ['imu_data','ego_pose']:
                for ind, member in enumerate(getattr(self, table)):
                    final_token = member['scene_token'] + '_' + member['token']
                    self._token2ind[table][final_token] = ind
            else:

                for ind, member in enumerate(getattr(self, table)):
                    self._token2ind[table][member['token']] = ind

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))


#-----------根据self._token2ind中token与index的对应关系索引出对应的记录----------------
#----------这里pose和imudata也要对应修改，加入scene_token----------------
    def get(self, table_name: str, token: str, scene_token=None) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :param scnen_token: Scene Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)
    
        return getattr(self, table_name)[self.getind(table_name, token, scene_token)]
    
#----------这里pose和imudata也要对应修改，加入scene_token----------------
    def getind(self, table_name: str, token: str, scene_token=None) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        if table_name in ['imu_data','ego_pose']:
            assert scene_token!=None, "scene_token is needed for imu_data and ego_pose"
            return self._token2ind[table_name][scene_token+'_'+token]
        
        return self._token2ind[table_name][token]
#--------------------------------------------------------------------------------

#-------------------获取返回lidar/ego下的标注信息,每个bbox以Box类存储----------------------------
#----------这里pose和imudata也要对应修改，加入scene_token----------------
    def get_annotation_box(self, sample_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('annotations', sample_token)['annotations']
        sync = self.get('sample_data', sample_token)
        scene_token = self.get('sample', sample_token)['scene_token']
        ego_pose = self.get('ego_pose', sync['ego_pose']['lidar_top_compensation'],scene_token)['pose'] 
        ego_pose = np.array(ego_pose).reshape(4, 4)
        global_to_ego = np.linalg.inv(ego_pose)
        box_list = []
        box_velocity_dict = self.box_velocity(sample_token)  
        for box in record:
            box_center = [box['center']['x'], box['center']['y'], box['center']['z']] 
            box_size = [box['size']['y'], box['size']['x'], box['size']['z']] 
            box_orientation = Quaternion(axis=[0, 0, 1], radians=box['rotation']['z']) 
            box_visibility = box['visibility']
            box_name = box['category']
            box_id = box['id']
            
            box_velocity_global = box_velocity_dict[box_id] 
            
            box_velocity_ego = np.matmul(box_velocity_global, global_to_ego[:3,:3].T)
             
            box_list.append(Box(center=box_center, size=box_size, orientation=box_orientation, \
                                velocity=box_velocity_ego, visibility=box_visibility,name=box_name, track_id=box_id))

        return box_list


    def box_velocity(self, sample_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """
        anno_velocity_dict = OrderedDict()
        prev_sample_token = self.get('sample',sample_token)['prev']
        next_sample_token = self.get('sample',sample_token)['next']
        
        if prev_sample_token != '':
            prev_anno_center = self.transform_anno_center_to_global(prev_sample_token)
        else:
            prev_anno_center = {}
        if next_sample_token != '':
            next_anno_center = self.transform_anno_center_to_global(next_sample_token)
        else:
            next_anno_center = {}
        current_anno_center = self.transform_anno_center_to_global(sample_token)

        
        
        for track_id, center_xyz in current_anno_center.items():
            max_time_diff_init = max_time_diff
            
            if track_id not in prev_anno_center and track_id not in next_anno_center:
                anno_velocity_dict[track_id] = np.array([np.nan, np.nan, np.nan])
                continue
            
            elif track_id in prev_anno_center and track_id not in next_anno_center:
                pos_first = np.array(prev_anno_center[track_id])
                pos_last = np.array(center_xyz)
                pos_diff = pos_last - pos_first
                time_first = 1e-6 * int(prev_sample_token) 
                time_last = 1e-6 * int(sample_token)
                time_diff = time_last - time_first
             
            elif track_id not in prev_anno_center and track_id in next_anno_center:
                pos_first = np.array(center_xyz)
                pos_last = np.array(next_anno_center[track_id])
                pos_diff = pos_last - pos_first
                time_first = 1e-6 * int(sample_token)
                time_last = 1e-6 * int(next_sample_token)
                time_diff = time_last - time_first
            
            elif track_id in prev_anno_center and track_id in next_anno_center:
                pos_first = np.array(prev_anno_center[track_id])
                pos_last = np.array(next_anno_center[track_id])
                pos_diff = pos_last - pos_first
                time_first = 1e-6 * int(prev_sample_token)
                time_last = 1e-6 * int(next_sample_token)
                time_diff = time_last - time_first

                max_time_diff_init *= 2


            if time_diff > max_time_diff_init:
                # If time_diff is too big, don't return an estimate.
                anno_velocity_dict[track_id] = np.array([np.nan, np.nan, np.nan])
            else:
                anno_velocity_dict[track_id] = pos_diff / time_diff
            
        return anno_velocity_dict


    
    def transform_anno_center_to_global(self, sample_token: str) -> dict:

        
        current_anno = self.get('annotations', sample_token)['annotations']
        current_sync = self.get('sample_data', sample_token)
        scene_token = self.get('sample', sample_token)['scene_token']
        current_pose = self.get('ego_pose', current_sync['ego_pose']['lidar_top_compensation'],scene_token)['pose'] #--16位pose
        current_pose = np.array(current_pose).reshape(4, 4)
        box_id = [] 
        boxes_center_ego = [] 
        for box in current_anno:
            box_center = [box['center']['x'], box['center']['y'], box['center']['z']] 
            boxes_center_ego.append(box_center)
            box_id.append(box['id'])
        boxes_center_ego = np.array(boxes_center_ego)
        boxes_center_global = transform_points(boxes_center_ego, current_pose) 
        anno_global_center = OrderedDict((id_val, center) for id_val, center in zip(box_id, boxes_center_global)) 
        return anno_global_center
    

if __name__ == '__main__':
    newsc = NewScenes(version='v1.0-mini', dataroot='data/newscenes-mini', verbose=True)
    print('i')
    
    
    
   