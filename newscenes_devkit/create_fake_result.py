import random
import numpy as np
import mmcv
from newscenes_devkit.newscenes import NewScenes
from os import path as osp




if __name__ == '__main__':
    input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True)

    newsc = NewScenes(version='v1.0-mini', dataroot='data/newscenes-mini', verbose=True)
    #----------读取每个sample，将验证集的拿出来----------
    sample_tokens_all = [s['token'] for s in newsc.sample]
    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = newsc.get('sample', sample_token)['scene_token']
        
        if scene_token in newsc.scene_split['val_mini']:
            sample_tokens.append(sample_token)
    #------------第一种，取出真值，置信度为1，生成假的测试json结果-----------------------------
    newsc_annos = {}
    for token in mmcv.track_iter_progress(sample_tokens):
        annos = []
        gt_boxes = newsc.get_annotation_box(token)
        for i, box in enumerate(gt_boxes):
            newsc_anno = dict(
                sample_token=token, #---token
                translation=box.center.tolist(), #---lidar/ego坐标系中心点
                size=box.wlh.tolist(), #---wlh
                rotation=box.orientation.elements.tolist(), #----四元数
                velocity=box.velocity[:2].tolist(), #----vxvy
                detection_name=box.name if box.name!='cyclist' else 'rider', #----name
                detection_score=1, #---置信度
                            )
            annos.append(newsc_anno)
        newsc_annos[token] = annos

    newsc_submission = {'meta': input_modality,'results': newsc_annos}
    current_path = osp.dirname(osp.abspath(__file__))
    res_path = osp.join(current_path, 'result_newsc_fakegt.json')
    print('Results writes to', res_path)
    mmcv.dump(newsc_submission, res_path)
    

    

