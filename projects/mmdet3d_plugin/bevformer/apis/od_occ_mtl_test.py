# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.

# ------------------------------------------------------
# Code by [TONGJI] [Lianqing Zheng]. All rights reserved.
# ------------------------------------------------------

import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from newscenes_devkit.newscenes import NewScenes


import mmcv
import numpy as np
import pycocotools.mask as mask_util


#-------------Custom rewrite of result handling---------------------
def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False,bad_condition_occ=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    bbox_results = []
    occ_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info() 
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))  #---0/
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    #-----------OCC bad weather evaluation-----------------
    if bad_condition_occ:
        newsc = NewScenes(version='v1.0-trainval', dataroot='data/NewScenes_Final', verbose=True)
        print("OCC恶劣天气评估!!!!!!!!!!!!!!!!!!!!!")
    for i, data in enumerate(data_loader):  #---fetch data----
        with torch.no_grad():
            #---result=
            result = model(return_loss=False, rescale=True, **data)  #--call forward; the 'rescale' has no effect here--
            #-----------OCC bad weather testing-----------------
            if bad_condition_occ:
                sample_token = data['img_metas'][0].data[0][0]['sample_idx']
                scene_token = newsc.get('sample', sample_token)['scene_token']
                scene_meta_dict = newsc.get('meta', scene_token)['meta']
                weather = scene_meta_dict['weather']
                lighting = scene_meta_dict['lighting']
                if not (weather == 'rainy' or lighting == 'night'):
                    del result['occ_results']
            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'occ_results' in result.keys():
                    occ_results.extend(result['occ_results'])
                    batch_size = len(result['occ_results'])
            else: 
                batch_size = len(result) #--1
                bbox_results.extend(result) 


        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    
    if gpu_collect:
        if 'bbox_results' in result.keys():
            bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if 'occ_results' in result.keys():
            occ_results = collect_results_gpu(occ_results, len(dataset))
        
    else:
        if 'bbox_results' in result.keys():
            bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        if 'occ_results' in result.keys():

            tmpdir = tmpdir+'_mask' if tmpdir is not None else None
            occ_results = collect_results_cpu(occ_results, len(dataset), tmpdir)

    return {'bbox_results': bbox_results, 'occ_results': occ_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace. 512 of them
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir (write each GPU's part result to a pkl file)
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()  #---Distributed synchronization barrier, wait for each process to finish--
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir (combine each process's results to generate the final result) ----
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size) 