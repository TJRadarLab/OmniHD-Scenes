"""
merge 3D detection dataset and occupancy gt data
"""
import mmcv 
import os
import tqdm

def merge(root_dir):
    
    for data_type in ['train', 'val']:
        ## 读取目标检测的pkl文件信息
        data_path = f'{root_dir}/newscenes-final_infos_temporal_{data_type}.pkl'
        data = mmcv.load(data_path)
        metadata = data['metadata']
        print('调试')
        data_infos = data['infos']
        save_infos = []
        for index in tqdm.tqdm(range(len(data_infos))):
            info = data_infos[index]
            occ_path = data['infos'][index]['lidar_path']
            occ_path = occ_path.replace('lidar/lidar_top_compensation', 'occ_gt')
            occ_path = occ_path.replace('bin', 'npz')
            info['occ_path'] = occ_path
            save_infos.append(info)

        save_path = os.path.join(root_dir, 'newscenes-final_infos_temporal_occ_{}.pkl'.format(data_type))
        save_data = dict(infos=save_infos, metadata = metadata)
        mmcv.dump(save_data, save_path)

if __name__ == '__main__':
    root_dir = './data/NewScenes_Final'
    merge(root_dir)