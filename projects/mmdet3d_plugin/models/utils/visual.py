# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Written by [TONGJI] [Lianqing Zheng]
# All rights reserved. Unauthorized distribution prohibited.
# Feel free to reach out for collaboration opportunities.
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2


def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor*255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    
    tensor = make_grid(tensor, pad_value=pad_value, normalize=False).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2


def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor * 255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=False).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


from skimage import io
import numpy as np
import mmcv


def featuremap_to_greymap_avgpool(feature_map):
    """
    feature_map: (C, sizey, sizex)
    grey_map: (sizey, sizex)
    """
    import torch
    import numpy as np
    import cv2
    if len(feature_map.shape) == 3:
        feature_map = feature_map.unsqueeze(dim=0)
    elif len(feature_map.shape) == 4:
        pass
    else:
        raise NotImplementedError
    channel_weights = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
    reduced_map = (channel_weights * feature_map).sum(dim=1).squeeze(dim=0)
    reduced_map = torch.relu(reduced_map)
    a_min = torch.min(reduced_map)
    a_max = torch.max(reduced_map)
    normed_map = (reduced_map - a_min) / (a_max - a_min)
    grey_map = normed_map
    return grey_map


def featuremap_to_greymap_maxpool(feature_map):
    """
    feature_map: (C, sizey, sizex)
    grey_map: (sizey, sizex)
    """
    import torch
    import numpy as np
    import cv2
    if len(feature_map.shape) == 3:
        feature_map = feature_map.unsqueeze(dim=0)
    elif len(feature_map.shape) == 4:
        pass
    else:
        raise NotImplementedError
    feature_map = torch.flip(feature_map, [2])
    max_values, _ = torch.max(feature_map, dim=1)
    max_values = max_values.squeeze(0)
    reduced_map = max_values
    a_min = torch.min(reduced_map)
    a_max = torch.max(reduced_map)
    normed_map = (reduced_map - a_min) / (a_max - a_min)
    grey_map = normed_map
    return grey_map


def greymap_to_rgbimg(map_grey, background=None, background_ratio=0.2, CHW_format=False):
    """
    map_grey: np, (sizey, sizex), values in 0-1
    background: np, (sizey, sizex, 3), values in 0-255.
    """
    map_img = plt.cm.viridis(map_grey)[..., :3]
    return map_img


def draw_bev(x, filename=None):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            plt.figure(figsize=(12, 8))
            plt.title('Channel-wise Maximized, ReLU and Normalized Feature Map')
            plt.imshow(rgb)
            plt.axis('off')
            plt.savefig(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/channel_maximized_feature_map_normalized_lssbev0901{i}.png', bbox_inches='tight')

            gray = featuremap_to_greymap_avgpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            plt.figure(figsize=(12, 8))
            plt.title('GPA, ReLU and Normalized Feature Map')
            plt.imshow(rgb)
            plt.axis('off')
            plt.savefig(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/avgpool_feature_map_normalized_lssbev00901{i}.png', bbox_inches='tight')

    if len(x.shape) == 3:
        gray = featuremap_to_greymap_maxpool(x)
        rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
        plt.figure(figsize=(12, 8))
        plt.title('Channel-wise Maximized, ReLU and Normalized Feature Map')
        plt.imshow(rgb)
        plt.axis('off')
        if filename is not None:
            plt.savefig(f'/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/max_historybev_{filename}.png', bbox_inches='tight')


def draw_bev_img(x, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            ori_img = image_metas[0]['filename'][0]
            current_img = mmcv.imread(ori_img)
            cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB, current_img)

            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.title('Channel-wise Maximized Feature Map')
            plt.imshow(rgb)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Original Image')
            plt.imshow(current_img)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_bev_img/{}_{}.png'.format(ori_img.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_bev_img_rcfusion(img, fuse, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(img.shape) == 4:
        for i in range(img.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(img[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            gray2 = featuremap_to_greymap_maxpool(fuse[i])
            rgb2 = greymap_to_rgbimg(gray2.cpu().detach().numpy())
            ori_img = image_metas[0]['filename'][0]
            current_img = mmcv.imread(ori_img)
            cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB, current_img)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.title('Original Image')
            plt.imshow(current_img)
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.title('img bev')
            plt.imshow(rgb)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title('fuse bev')
            plt.imshow(rgb2)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_rcfusion_img/{}_{}.png'.format(ori_img.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_bevformer_bev_img(x, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            ori_img = image_metas[0]['filename'][0]
            current_img = mmcv.imread(ori_img)
            cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB, current_img)

            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.title('Channel-wise Maximized Feature Map')
            plt.imshow(rgb)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Original Image')
            plt.imshow(current_img)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_bevformer_bev_img/{}_{}.png'.format(ori_img.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_lss_img_depth_bev(x, img_metas, vis_time_bev, min_depth, predict_depth):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            if i != 0:
                continue
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            ori_img_front = image_metas[0]['filename'][0]
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)
            ori_img_back = image_metas[0]['filename'][3]
            img_back = mmcv.imread(ori_img_back)
            cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB, img_back)
            min_depth_filter = min_depth
            min_depth_filter = torch.clamp(min_depth, min=1, max=59)
            gt_depth_front = min_depth_filter[0].cpu().numpy()
            gt_depth_back = min_depth_filter[3].cpu().numpy()
            if predict_depth.shape[2] == 118:
                predict_depth = predict_depth.flatten(0, 1).permute(0, 2, 3, 1).argmax(axis=-1) * 0.5 + 1
            else:
                predict_depth = predict_depth.flatten(0, 1).permute(0, 2, 3, 1).argmax(axis=-1) + 1
            pred_depth_front = predict_depth[0].cpu().numpy()
            pred_depth_back = predict_depth[3].cpu().numpy()

            plt.figure(figsize=(12, 8))
            plt.subplot(3, 3, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(3, 3, 2)
            plt.title('Front View - Ground Truth Depth')
            plt.imshow(gt_depth_front, cmap='viridis')
            plt.colorbar(label='Depth (m)')
            plt.axis('off')

            plt.subplot(3, 3, 3)
            plt.title('Front View - Predicted Depth')
            plt.imshow(pred_depth_front, cmap='viridis')
            plt.colorbar(label='Depth (m)')
            plt.axis('off')

            plt.subplot(3, 3, 4)
            plt.title('Back View - RGB Image')
            plt.imshow(img_back)
            plt.axis('off')

            plt.subplot(3, 3, 5)
            plt.title('Back View - Ground Truth Depth')
            plt.imshow(gt_depth_back, cmap='viridis')
            plt.colorbar(label='Depth (m)')
            plt.axis('off')

            plt.subplot(3, 3, 6)
            plt.title('Back View - Predicted Depth')
            plt.imshow(pred_depth_back, cmap='viridis')
            plt.colorbar(label='Depth (m)')
            plt.axis('off')

            plt.subplot(3, 3, 7)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_lss_bev_depth/{}_{}.png'.format(ori_img_front.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_rcdetsoc_img_bev(x, bev_seg, bev_seg_gt, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())

            bev_seg = torch.flip(bev_seg, [2])
            bev_mask = bev_seg[i].squeeze().cpu().detach().numpy()

            bev_seg_gt = torch.flip(bev_seg_gt, [2])
            bev_mask_gt = bev_seg_gt[i].squeeze().cpu().detach().numpy()
            ori_img_front = image_metas[0]['filename'][0]
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)
            ori_img_back = image_metas[0]['filename'][3]
            img_back = mmcv.imread(ori_img_back)
            cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB, img_back)

            plt.figure(figsize=(12, 8))
            plt.subplot(3, 2, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(3, 2, 2)
            plt.title('Back View - RGB Image')
            plt.imshow(img_back)
            plt.axis('off')

            plt.subplot(3, 2, 3)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')

            plt.subplot(3, 2, 4)
            plt.title('BEV_mask')
            plt.imshow(bev_mask)
            plt.axis('off')
            plt.subplot(3, 2, 5)
            plt.title('BEV_mask_gt')
            plt.imshow(bev_mask_gt)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_rcdetsoc_bev_seg/{}_{}.png'.format(ori_img_front.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_bevfusion_img_bev(x, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            ori_img_front = image_metas[0]['filename'][0]
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)
            ori_img_back = image_metas[0]['filename'][3]
            img_back = mmcv.imread(ori_img_back)
            cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB, img_back)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.title('Back View - RGB Image')
            plt.imshow(img_back)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_bevfusion_bev/{}_{}.png'.format(ori_img_front.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_vod(x, bev_seg, bev_seg_gt, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())

            bev_seg = torch.flip(bev_seg, [2])
            bev_mask = bev_seg[i].squeeze().cpu().detach().numpy()

            bev_seg_gt = torch.flip(bev_seg_gt, [2])
            bev_mask_gt = bev_seg_gt[i].squeeze().cpu().detach().numpy()
            ori_img_front = image_metas[0]['filename']
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title('BEV_mask')
            plt.imshow(bev_mask)
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.title('BEV_mask_gt')
            plt.imshow(bev_mask_gt)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_vod/{}_{}.png'.format(ori_img_front.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_tj4d(x, bev_seg, bev_seg_gt, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i][:, 30:-30, :])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())

            bev_seg = torch.flip(bev_seg, [2])
            bev_mask = bev_seg[i].squeeze().cpu().detach().numpy()

            bev_seg_gt = torch.flip(bev_seg_gt, [2])
            bev_mask_gt = bev_seg_gt[i].squeeze().cpu().detach().numpy()
            ori_img_front = image_metas[0]['filename']
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title('BEV_mask')
            plt.imshow(bev_mask)
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.title('BEV_mask_gt')
            plt.imshow(bev_mask_gt)
            plt.axis('off')
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/draw_tj4d/{}_{}.png'.format(ori_img_front.split('/')[-1][:-4], vis_time_bev), bbox_inches='tight')
            plt.close()


def draw_doracamom(x, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            center_x = rgb.shape[1] // 2
            center_y = rgb.shape[0] // 2

            ori_img_front = image_metas[0]['filename'][0]
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)
            ori_img_back = image_metas[0]['filename'][3]
            img_back = mmcv.imread(ori_img_back)
            cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB, img_back)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.title('Back View - RGB Image')
            plt.imshow(img_back)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')
            plt.plot(center_x, center_y, 'rx', markersize=4)
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/Doracamom_final_featuremap/{}_feature.png'.format(img_metas[0]['sample_idx']), bbox_inches='tight')
            plt.close()


def draw_bevformer_test(x, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            center_x = rgb.shape[1] // 2
            center_y = rgb.shape[0] // 2

            ori_img_front = image_metas[0]['filename'][0]
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)
            ori_img_back = image_metas[0]['filename'][3]
            img_back = mmcv.imread(ori_img_back)
            cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB, img_back)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.title('Back View - RGB Image')
            plt.imshow(img_back)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')
            plt.plot(center_x, center_y, 'rx', markersize=4)
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/bevformer_T_final_featuremap/{}_feature.png'.format(img_metas[0]['sample_idx']), bbox_inches='tight')
            plt.close()


def draw_bevfusion_test(x, img_metas, vis_time_bev):
    import matplotlib.pyplot as plt
    if len(x.shape) == 4:
        for i in range(x.shape[0]):
            image_metas = img_metas.copy()
            gray = featuremap_to_greymap_maxpool(x[i])
            rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            center_x = rgb.shape[1] // 2
            center_y = rgb.shape[0] // 2

            ori_img_front = image_metas[0]['filename'][0]
            img_front = mmcv.imread(ori_img_front)
            cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB, img_front)
            ori_img_back = image_metas[0]['filename'][3]
            img_back = mmcv.imread(ori_img_back)
            cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB, img_back)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.title('Front View - RGB Image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.title('Back View - RGB Image')
            plt.imshow(img_back)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title('BEV_map')
            plt.imshow(rgb)
            plt.axis('off')
            plt.plot(center_x, center_y, 'rx', markersize=4)
            plt.savefig('/mnt/zhenglianqing/bevformer_noted/debug_some_imgresult/BEVFusion_final_featuremap/{}_feature.png'.format(img_metas[0]['sample_idx']), bbox_inches='tight')
            plt.close()