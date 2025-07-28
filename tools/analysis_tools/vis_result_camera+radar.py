"""
bbox投影到六路相机生成视频 + 俯视雷达bbox 
"""
import json
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import matplotlib.pyplot as plt
import pickle
from newscenes_devkit.newscenes import NewScenes
from newscenes_devkit.eval.common.loaders import load_prediction, load_gt, filter_eval_boxes
from newscenes_devkit.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from newscenes_devkit.eval.detection.config import config_factory
from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadRadarPointsMultiSweeps
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
from newscenes_devkit.eval.common.utils import quaternion_yaw, Quaternion


point_cloud_range = [-60,-40,-3,60,40,5] #----目标范围xyz
cameras_flag = ['camera_left_front',
                'camera_front', 
                'camera_right_front',
                'camera_left_back', 
                'camera_back',
                'camera_right_back',  
                ]

class_names = ['car', 'pedestrian', 'rider', 'large_vehicle']
detection_mapping = {
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
            } #---后期在修改


def custom_colors(type='bgr'):

    colors = []
    if type == 'rgb':
        colors.append([0, 255, 255])     # yellow  #00ffff
        colors.append([245, 135, 56])    # light blue  #f58738
        colors.append([0, 255, 0])       # green  #00ff00
        colors.append([255, 0, 255])     # magenta  #ff00ff
        colors.append([240, 32, 160])    # purple  #f020a0
        colors.append([255, 255, 0])     # cyan  #ffff00
        colors.append([0, 0, 255])       # red  #0000ff
        colors.append([0, 215, 255])     # gold  #00d7ff
        colors.append([144, 238, 144])   # light green  #90ee90
        colors.append([128, 0, 0])       # navy  #800000
        colors.append([0, 0, 128])       # maroon  #000080
        colors.append([255, 0, 0])       # blue  #ff0000
        colors.append([128, 128, 0])     # teal  #808000
        colors.append([0, 128, 128])     # olive  #008080
        colors.append([128, 0, 0])       # navy  #800000
    elif type == 'bgr':#--------看不清楚-----------------
        # colors.append([255, 255,0])     # yellow  #00ffff
        # colors.append([56, 135, 245])    # light blue  #f58738
        # colors.append([0, 255, 0])       # green  #00ff00
        # colors.append([255, 0, 255])     # magenta  #ff00ff
        
        colors.append([0, 0, 255])       # blue  #ff0000
        colors.append([0, 0, 255])       # blue  #ff0000
        colors.append([0, 0, 255])       # blue  #ff0000
        colors.append([0, 0, 255])       # blue  #ff0000


    return colors

# 读取json文件
def read_json(path):

    # 读取文件内容到字符串中  
    with open(path, 'r', encoding='utf-8') as file:  
        json_str = file.read()  
    # 使用json.loads()方法解析JSON字符串  
    data = json.loads(json_str) 

    return data


# 读取图片
def cv_imread(file_path):
    #--------保持原始图像的颜色通道数和位深度不变，输出BGR格式----
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1) 
    # im decode读取的是bgr
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img


# 读取bin文件
def read_bin_lidar_point(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    points = data.reshape(-1, 6) # 激光雷达点云（六维特征）
    return points[:,:4]

# 读取单个radar点云
def read_bin_float32_radar_point(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    points = data.reshape(-1, 8) # 毫米波雷达点云（八维特征)暂时都用
    return points


# 读取100Hz的pose文件
def read_poses_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    ego_poses = {}
    for line in lines:
        data = line.strip().split(',')
        matrix = np.array(data[1:]).astype(float).reshape(4, 4)
        ego_poses[str(int(data[0][:16]))] = matrix
        
    return ego_poses


# 读取所有radar点，并转到每个radar时刻ego下
def read_bin_all_radar_point(radar_path,radar_file_dict,radars2ego):
    
    all_radar_points = {}
    for radar_name, file_name in radar_file_dict.items():
        radar_file_path = os.path.join(radar_path,radar_name,file_name+'.bin')
        radar_point = read_bin_float32_radar_point(radar_file_path)
        radar_point_ego = transform_points(radar_point, radars2ego[radar_name])
        all_radar_points[radar_name] = radar_point_ego
    return all_radar_points


def get_calib_radar(calib_data):

    radars2ego = {}
    radars2ego['radar_left_front'] = np.array(calib_data['radar_left_front']['radar2ego'],dtype=np.float32)
    radars2ego['radar_front'] = np.array(calib_data['radar_front']['radar2ego'],dtype=np.float32)
    radars2ego['radar_right_front'] = np.array(calib_data['radar_right_front']['radar2ego'],dtype=np.float32)
    radars2ego['radar_left_back'] = np.array(calib_data['radar_left_back']['radar2ego'],dtype=np.float32)
    radars2ego['radar_back'] = np.array(calib_data['radar_back']['radar2ego'],dtype=np.float32)
    radars2ego['radar_right_back'] = np.array(calib_data['radar_right_back']['radar2ego'],dtype=np.float32)
    
    return radars2ego 


# 补偿radar点云的pose到lidar时刻ego下
def compensation_radar_points_multisweep(radar_six_points, ego_pose_every_sensor_timestamp,ego_pose_file,lidar_pose_timestamp):
    radar_six_new_points = {}

    for  sensor_name, pose_timestamp in ego_pose_every_sensor_timestamp.items():
        if sensor_name == 'lidar_top_compensation':
            continue
        radar_in_global = transform_points(radar_six_points[sensor_name], ego_pose_file[pose_timestamp])
        radar_in_new_ego = transform_points(radar_in_global, np.linalg.inv(ego_pose_file[(lidar_pose_timestamp)]))
        radar_six_new_points[sensor_name] = radar_in_new_ego.astype(np.float32)

    return radar_six_new_points


# 点云变换
def transform_points(point_cloud, transformation_matrix):

    # 旋转变换
    transformed_xyz = np.matmul(point_cloud[:, :3], transformation_matrix[:3,:3].T)
    # 平移变化
    transformed_xyz[:,:3] += transformation_matrix[:3, 3]
    # 添加点云的intensity维度
    transformed_points = np.hstack((transformed_xyz[:, :3], point_cloud[:, 3:]))

    return transformed_points


# 读取标注json文件center和size并计算单个相机视角下3Dbbox的八个角坐标
def bbox_annotations_2d(result_data, cam_intric, ego_to_cam, height, width):
    '''
    result_data:标注信息的dict组成的list
    cam_intric_list:单个相机的内参矩阵列表
    one_cam_corners_img:单个相机视角下3Dbbox的八个角坐标
    '''
    #print("-----------------------------------")
    #print(height, width)
    one_cam_corners_img = []
    one_cam_project_category = []
    for detection_data in result_data:
        category = detection_data.detection_name
        score = detection_data.detection_score
        bbox_size_3d = detection_data.size
        rotation_3d = detection_data.rotation
        velocity = detection_data.velocity
        center_point_3d = detection_data.translation

        if score < 0.15:
            pass
        else:
            one_cam_project_category.append(category)

            x = center_point_3d[0]
            y = center_point_3d[1]
            z = center_point_3d[2]

            w = bbox_size_3d[0]
            l = bbox_size_3d[1]
            h = bbox_size_3d[2]
            
            #bbox_size_3d = np.array([l, h, w])
            #----------这里长宽高的顺序是有讲究的，不要随便改----------
            x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2 ,-h/2]
            corners_3d = np.vstack([x_corners, y_corners, z_corners])
            #------------------这里要确定一下-------------------------
            # rot = 2 * np.arccos(rotation_3d[3])
            rot = quaternion_yaw(Quaternion(rotation_3d))
            # rot1 = 2 * np.arccos(rotation_3d[3]) + np.pi

            R = np.array([
                    [np.cos(rot), -np.sin(rot), 0],
                    [np.sin(rot), np.cos(rot), 0],
                    [0, 0, 1]
                ])
            # 将旋转角度转换为弧度
            corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])


            # 从ego坐标系转到相机坐标系
            corners_cam = transform_points(corners_3d, ego_to_cam)
            # 投影到图像坐标系
            corners_img = np.matmul(corners_cam, cam_intric.T)
            corners_img[:,0] = corners_img[:,0] / corners_img[:,2]
            corners_img[:,1] = corners_img[:,1] / corners_img[:,2]
            
            # 剔除图像外的点
            mask = []
            for j in range(len(corners_img)):
                if (0 <= corners_img[j, 0] <= width) & (0 <= corners_img[j, 1] <= height) & (0 <= corners_img[j, 2]):
                    mask.append(1)
                else:
                    mask.append(0)
            mask = np.array(mask)

            #mask = (-10 <= (corners_img[:, 0])) & ((corners_img[:, 0]) <= width+10) \
            #    & (-10 <= (corners_img[:, 1])) & ((corners_img[:, 1]) <= height+10) \
            #    & (0 <= (corners_img[:, 2]))

            n = np.sum(mask == 1)
            if n >= 4 :
                one_cam_corners_img.append(corners_img)
            else:
                one_cam_corners_img.append(corners_img[mask == 1])
            
            #print(corners_img)

    return one_cam_corners_img, one_cam_project_category


# 画bbox
def draw_bbox_corners(one_cam_corners_img, one_cam_project_category, img, thickness):

    for i in range(len(one_cam_corners_img)):
        corners = one_cam_corners_img[i]
        corners = corners.astype(int)
        category = one_cam_project_category[i]
        colors = custom_colors(type='rgb')
        bbox_color = colors[class_names.index(category)]
        #bbox_color = colors[class_names.index(detection_mapping[category])]
        bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])
        if corners.shape == (8,3):
            #print(corners)
            cv2.line(img, (corners[0, 0], corners[0, 1]),
                (corners[1, 0], corners[1, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[1, 0], corners[1, 1]),
                (corners[2, 0], corners[2, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[2, 0], corners[2, 1]),
                (corners[3, 0], corners[3, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[3, 0], corners[3, 1]),
                (corners[0, 0], corners[0, 1]), color=bbox_color, thickness=thickness)
            
            cv2.line(img, (corners[4, 0], corners[4, 1]),
                (corners[5, 0], corners[5, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[5, 0], corners[5, 1]),
                (corners[6, 0], corners[6, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[6, 0], corners[6, 1]),
                (corners[7, 0], corners[7, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[7, 0], corners[7, 1]),
                (corners[4, 0], corners[4, 1]), color=bbox_color, thickness=thickness)
            
            cv2.line(img, (corners[0, 0], corners[0, 1]),
                (corners[4, 0], corners[4, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[1, 0], corners[1, 1]),
                (corners[5, 0], corners[5, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[2, 0], corners[2, 1]),
                (corners[6, 0], corners[6, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[3, 0], corners[3, 1]),
                (corners[7, 0], corners[7, 1]), color=bbox_color, thickness=thickness)
                
            cv2.line(img, (corners[0, 0], corners[0, 1]),
                (corners[5, 0], corners[5, 1]), color=bbox_color, thickness=thickness)
            cv2.line(img, (corners[1, 0], corners[1, 1]),
                (corners[4, 0], corners[4, 1]), color=bbox_color, thickness=thickness)
        else:
            pass

    return img

# lidar点多帧叠加
def sweep_lidar_points(sync_files_path, lidar_path, radar_path, pose_file, index, multi_sweep_radar, radars2ego):

    #--------当前的sync---------
    sync_dict = read_json(sync_files_path[index])

    # lidar时刻下ego下的lidar点及pose时间戳
    lidar_file_path = os.path.join(lidar_path, sync_dict['lidar']['lidar_top_compensation']+'.bin')
    lidar_points = read_bin_lidar_point(lidar_file_path)
    lidar_pose_timestamp =  sync_dict['ego_pose']['lidar_top_compensation']

    # 多帧叠加
    ego_pose_file = read_poses_from_file(pose_file)
    radar_sweeps = []
    for idx in range(max(0, index-multi_sweep_radar+1), index+1):
        sync_dict_idx = read_json(sync_files_path[idx])
        ego_pose_every_sensor_timestamp = sync_dict_idx['ego_pose']
        radar_file_dict = sync_dict_idx['radars']
        radar_six_points = read_bin_all_radar_point(radar_path, radar_file_dict, radars2ego)
        radar_six_points = compensation_radar_points_multisweep(radar_six_points, ego_pose_every_sensor_timestamp,ego_pose_file,lidar_pose_timestamp)
        radar_sweeps.append(radar_six_points)

    return lidar_points, radar_sweeps


# 可视化点云和bbox
def vis_lidar_radar_bbox(lidar_points, radar_sweeps, annotations_data,save_path):

    lidar_x, lidar_y, lidar_z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
    z_lidar = lidar_points[:, 2]


    radar_x, radar_y, radar_z = radar_sweeps[:, 0], radar_sweeps[:, 1], radar_sweeps[:, 2]
    z_radar = radar_sweeps[:, 2] 

    plt.figure(figsize=(12, 8))
    

    #ax = fig.add_subplot(111, projection='3d')
    plt.scatter(lidar_x, lidar_y, c='k', s=0.2, alpha=0.5)
    plt.scatter(radar_x, radar_y, c='r', s=15, alpha=0.5)

    # 绘制bbox
    for detection_data in annotations_data:
        category = detection_data.detection_name
        score = detection_data.detection_score
        bbox_size_3d = detection_data.size
        rotation_3d = detection_data.rotation
        velocity = detection_data.velocity
        center_point_3d = detection_data.translation
        visibility = detection_data.visibility
    
        
        if category == 'no_object':
            pass
        if score < 0.15:
            pass
        else:
            if visibility == 0:
                pass
            else:
                
                colors = custom_colors(type='bgr')
                bbox_color = np.array(colors[class_names.index(category)]) / 255.0
                x = center_point_3d[0]
                y = center_point_3d[1]
                z = center_point_3d[2]

                w = bbox_size_3d[0]
                l = bbox_size_3d[1]
                h = bbox_size_3d[2]

                # 右上角 顺时针 先上后下
                x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2 ,-h/2]
                corners_3d = np.vstack([x_corners, y_corners, z_corners])
                #-----------这里要确定一下--------------------
                #rot = 2 * np.arccos(rotation_3d[3])
                # rot = 2 * np.arccos(rotation_3d[3]) + np.pi
                rot = quaternion_yaw(Quaternion(rotation_3d))
                R = np.array([
                        [np.cos(rot), -np.sin(rot), 0],
                        [np.sin(rot), np.cos(rot), 0],
                        [0, 0, 1]
                    ])
                
                # 将旋转角度转换为弧度
                corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

                # 绘制边界框的4条边
                bbox_lines = [
                    (0,1), (1,2), (2,3), (3,0)
                ]

                for p1_idx, p2_idx in bbox_lines:
                # 绘制框
                    plt.plot([corners_3d[p1_idx, 0], corners_3d[p2_idx, 0]],
                            [corners_3d[p1_idx, 1], corners_3d[p2_idx, 1]],
                            color=bbox_color)
                
   

    plt.xlim(-60, 60)
    plt.ylim(-40, 40)
    plt.axis('off')
    # #plt.show()
    plt.savefig(save_path,bbox_inches='tight',dpi=300)
    plt.close()


if __name__ == '__main__':
    root_path = 'data/NewScenes_Final/'
    result_json_path = 'work_dirs/doracamom_20241120/final/Doracamom_1120_final_4frame_3encoder/val_result/Fri_Nov_29_11_03_03_2024/pts_bbox/results_newsc.json'
    version = 'v1.0-trainval'
    save_path = '/mnt/zlq/bevformer_noted/debug_some_imgresult/Doracamom_final_OD_points'
    save_img_path = '/mnt/zlq/bevformer_noted/debug_some_imgresult/Doracamom_final_OD_yaw'
    newsc = NewScenes(version=version, dataroot=root_path, verbose=True)

    config = config_factory('detection_newsc_config_final')

    pred_boxes, meta = load_prediction(result_json_path, config.max_boxes_per_sample, DetectionBox, verbose=True)

    pred_boxes = filter_eval_boxes(newsc, pred_boxes, config.class_range, verbose=True)

    val_pkl_path = 'data/NewScenes_Final/newscenes-final_infos_temporal_occ_val.pkl'
    f = open(val_pkl_path, 'rb')
    val_data = pickle.load(f)
    val_info = val_data['infos']

    start = 0
    for frame_info in val_info:
        token = frame_info['token']
        lidar_path = frame_info['lidar_path']
        cams = frame_info['cams']
        lidar2ego_R = frame_info['lidar2ego_rotation']
        lidar2ego_T = frame_info['lidar2ego_translation']
        
        scene_token = newsc.get('sample', token)['scene_token']
        
        # 检测结果
        results = []
        while start < len(pred_boxes.all):
            if pred_boxes.all[start].sample_token == token:
                result = pred_boxes.all[start]
                results.append(result)
                start += 1
            else:
                break

        # 根据sync文件取对应的img 
        camera_images = []
        camera_intrics = []
        ego_2_camera = []
        img_height= []
        img_width = []

        # 读取6张图片
        for current_camera in cameras_flag:
            img_path = cams[current_camera]['data_path']
            img_origin = cv_imread(img_path)

            height, width, _ = img_origin.shape
            cam_intric = cams[current_camera]['cam_intrinsic']
            cam_intric = np.array(cam_intric,dtype=np.float32)
            cam_distortion = cams[current_camera]['cam_distortion']
            cam_distortion = np.array(cam_distortion,dtype=np.float32)
            
            cam_to_ego = newsc.get('sensor_calibration',scene_token)['calib'][current_camera]['camera2ego']
            cam_to_ego = np.array(cam_to_ego,dtype=np.float32)
            ego_to_cam = np.linalg.inv(cam_to_ego)
            
            img_undistortion = cv2.undistort(img_origin,cam_intric, cam_distortion, None, cam_intric)

            camera_intrics.append(cam_intric)
            camera_images.append(img_undistortion)
            ego_2_camera.append(ego_to_cam)
            img_height.append(height)
            img_width.append(width)

        # 创建索引表
        order = [1, 5, 4, 2, 6, 3]
        # 绘图 包含6个子图的图形 
        fig, axs = plt.subplots(2, 3, figsize=(20, 8))
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.1)
        thickness = 3
        # 遍历六张图的数据和子图
        for j, (ax) in enumerate(axs.flatten()):
            id = order[j]
            one_cam_corners_img, one_cam_project_category = bbox_annotations_2d(results, camera_intrics[j], ego_2_camera[j], img_height[j], img_width[j])
            img = draw_bbox_corners(one_cam_corners_img, one_cam_project_category, camera_images[j], thickness)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 根据需要设置子图的标题
            #ax.set_title(f'Camera {id}')
            ax.axis('off')
            ax.margins(0)

        #显示所有的图'1693724597500491'
        img_save_path = os.path.join(save_img_path, token + '_image.png')
        plt.savefig(img_save_path, bbox_inches='tight',dpi=100)
        plt.close(fig)
    # #------------绘制点云和box--------------
    #     input_dict = dict(
    #         sample_idx=token,
    #         pts_filename=lidar_path,
    #         radars=frame_info['radars'],
    #     )
    #     lidar_loader = LoadPointsFromFile(coord_type='LIDAR',load_dim=6,use_dim=4)
    #     input_dict = lidar_loader(input_dict)
    #     lidar_points_mask = input_dict['points'].in_range_3d(point_cloud_range)
    #     lidar_points = input_dict['points'][lidar_points_mask].tensor.numpy()
    #     radar_loader = LoadRadarPointsMultiSweeps(load_dim=8,
    #              use_dim=[0, 1, 2, 3, 4, 5, 6, 7],
    #              sweeps_num=3, 
    #              file_client_args=dict(backend='disk'),
    #              max_num=300,
    #              pc_range=point_cloud_range, 
    #              test_mode=False)     
    #     input_dict = radar_loader(input_dict)
    #     radar_points = input_dict['points'].tensor.numpy()
    #     points_save_path = os.path.join(save_path, token + '_lidarradar.png')
    #     vis_lidar_radar_bbox(lidar_points, radar_points, results,points_save_path)