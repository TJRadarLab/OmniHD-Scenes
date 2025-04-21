<div align="center">

# OmniHD-Scenes: A Next-Generation Multimodal Dataset for Autonomous Driving

<sup>1, \*</sup>Lianqing Zheng, <sup>1, \*</sup>Long Yang, <sup>2, \*</sup>Qunshu Lin, <sup>1</sup>Wenjin Ai, <sup>3</sup>Minghao Liu, <sup>1</sup>Shouyi Lu, <sup>4</sup>Jianan Liu, <sup>1</sup>Hongze Ren, <sup>1</sup>Jingyue Mo, <sup>2</sup>Xiaokai Bai,<sup>5</sup>Jie Bai,<sup>1, †</sup>Zhixing Ma,<sup>1,#</sup>Xichan Zhu

<sup>1</sup>Tongji University, <sup>2</sup>Zhejiang University, <sup>3</sup>2077AI  
<sup>4</sup>Momoni AI
<sup>5</sup>Hangzhou City University

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2412.10734" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv:2412.10734-blue">
  </a>
  <a href="https://arxiv.org/abs/2412.10734" target='_blank'>
    <img src="https://img.shields.io/badge/Paper📖-blue">
  </a> 
  <a href="https://www.2077ai.com/OmniHD-Scenes" target='_blank'>
    <img src="https://img.shields.io/badge/Project🚀-blue">
  </a>
  <a href="https://github.com/TJRadarLab/OmniHD-Scenes" target='_blank'>
    <img src="https://img.shields.io/badge/Code🤖-blue">
  </a>
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE" target='_blank'>
  <img src="https://img.shields.io/badge/Code%20License:Apache%202.0-green.svg">
  </a>
</p>

## 🔥 News

•	**[2025-04-15]** 🎉 Our OmniHD-Scenes dataset v1.0 (~1.3TB) is openly [accessible](https://www.2077ai.com/contact) for research purposes.

•	**[2024-12-31]** 🌐 The [project page](https://www.2077ai.com/OmniHD-Scenes) is now online.

##  🛠️ Abstract
 We present OmniHD-Scenes, a large-scale multimodal dataset that provides comprehensive omnidirectional high-definition data. The OmniHD-Scenes dataset combines data from 128-beam LiDAR, six cameras, and six 4D imaging radar systems to achieve full environmental perception. To date, we have annotated 200 clips with more than 514K precise 3D bounding boxes. These clips also include semantic segmentation annotations for static scene elements. Alongside the proposed dataset, we establish comprehensive evaluation metrics, baseline models, and benchmarks for 3D detection and semantic occupancy prediction. 
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="Figs/vehicle.jpg" width="280"/>
  <img src="Figs/coordinate.jpg" width="280"/>
</div>
<div align="center">
  <b>Data Acquisition Platform and Coordinate System</b>
</div>


## ⚙️ Dataset Structure
OmniHD-Scenes is structured in clips, drawing inspiration from nuScenes' data composition format. The dataset is organized as follows.
```shell
OmniHD-Scenes
├── 1693714828633418               # Clip Scene
│   ├── cameras                    # Six Cameras
│   │   ├── camera_back                
│   │   │   ├── xxxxxxxxx.jpg         
│   │   │   └── ... 
│   │   ├── camera_front                
│   │   │   ├── xxxxxxxxx.jpg          
│   │   │   └── ... 
│   │   ├──...
│   │   ├── camera_right_front
│   │   │   ├── xxxxxxxxx.jpg          
│   │   │   └── ...                
│   ├── lidar                    # LiDAR
│   │   ├── lidar_top_compensation             
│   │   │   ├── xxxxxxxxx.bin          
│   │   │   └── ... 
│   ├── radars                    # Six 4D Radars
│   │   ├── radar_back                
│   │   │   ├── xxxxxxxxx.bin         
│   │   │   └── ... 
│   │   ├── radar_front                
│   │   │   ├── xxxxxxxxx.bin          
│   │   │   └── ... 
│   │   ├──...
│   │   ├── radar_right_front
│   │   │   ├── xxxxxxxxx.bin          
│   │   │   └── ...
├──...
├── 1693922406733409               
├── v1.0-trainval
│   ├── annotations.json           # 3D box label
│   ├── ego_pose.json                # ego pose
│   ├── imu.json                   # ego status
│   ├── meta.json                   # scene description
│   ├── sample_data.json           # index of all frames
│   ├── sample.json                # key frames
│   ├── scene_split.json           # train/test split
│   └── sensor_calibration.json    # calib parameters
```


<div align="center">
  <img src="Figs/3DOD.jpg" alt="logo" >
</div>
<div align="center">
  <b> Multiple scenes and 3D annotation visualization
  System</b>
</div>

<div align="center">
  <img src="Figs/OCC.jpg" alt="logo" >
</div>
<div align="center">
  <b> Multiple scenes and semantic occupancy visualization</b>
</div>


## 🔨 Quick Start
### Installation
You can install the whole repository by following these steps:

Clone
```
git clone https://github.com/TJRadarLab/OmniHD-Scenes.git
```
Create environment 
```
conda create -n omnihd python=3.8 -y
conda activate omnihd
```
Install pytorch
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Install mmcv/mmdet/mmseg
```
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```
Install mmdet3d
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 
pip install -v -e .  
```

### Test

Test a baseline model
```
./tools/dist_test.sh ./projects/configs/XXX/XXX.py ./work_dirs/XXX/XXX.pth 2
```

## 🍁 Baseline Results

![OD](./Figs/OD_results.png)

![OCC](./Figs/OCC_results.png)


## ⏳ To Do

- [ ] Release baseline models
- [ ] Release the OCC label
- [ ] Release the codebase (will be released after peer review)

## ⭐ Others
If you have any questions about the dataset, feel free to cantact us with tjradarlab@163.com & contact@2077.com.

## 😙 Acknowledgement

Many thanks to these exceptional open source projects:
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion)

As it is not possible to list all the projects of the reference papers. If you find we leave out your repo, please contact us and we'll update the lists.

## 📃Citation

```
@article{zheng2024omnihd,
  title={OmniHD-Scenes: A next-generation multimodal dataset for autonomous driving},
  author={Zheng, Lianqing and Yang, Long and Lin, Qunshu and Ai, Wenjin and Liu, Minghao and Lu, Shouyi and Liu, Jianan and Ren, Hongze and Mo, Jingyue and Bai, Xiaokai and others},
  journal={arXiv preprint arXiv:2412.10734},
  year={2024}
}
```