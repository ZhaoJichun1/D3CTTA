# **D^3CTTA: Domain-Dependent Decorrelation for Continual Test-Time Adaption of 3D LiDAR Segmentation[CVPR2025]**



## Abstract

Adapting pre-trained LiDAR segmentation models to dynamic domain shifts during testing is of paramount importance for the safety of autonomous driving. 
Most existing methods neglect the influence of domain changes and point density on the continual test-time adaption (CTTA) and require backpropagation and large batch sizes for stable adaption.
We approach this problem with three insights: 1) Point clouds at different distances usually have different densities resulting in distribution disparities; 2) The feature distribution of different domains varies, and domain-aware parameters can alleviate domain gaps; 3) Features are highly correlated and make segmentation of different labels confusing. 
To this end, this work presents D^3CTTA, an online backpropagation-free framework for 3D continual test-time adaption for LiDAR segmentation.
D^3CTTA consists of a distance-aware prototype learning module to integrate LiDAR-based geometry prior and a domain-dependent decorrelation module to reduce feature correlations among different domains and different categories.
Extensive experiments on three benchmarks showcase that our method achieves a state-of-the-art performance compared to both backpropagation-based methods and backpropagation-free methods.

## Installation
The code has been tested with Python 3.8, CUDA 11.1, pytorch 1.8.0 and pytorch-lighting 1.4.1.
Any other version may require to update the code for compatibility.

### Pip/Venv/Conda
In your virtual environment follow [MinkowskiEnginge](https://github.com/NVIDIA/MinkowskiEngine).
This will install all the base packages.

Additionally, you need to install:
- [open3d 0.13.0](http://www.open3d.org)
- [KNN-CUDA](https://github.com/unlimblue/KNN_CUDA)
- [pytorch-lighting 1.4.1](https://www.pytorchlightning.ai)
- tqdm
- pickle

If you want to work on nuScenes you need to install
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)


## Data preparation


### SemanticKITTI-C
prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
        ├── beam_missing/           
        │   ├── heavy/	
        |   |	   ├── labels
        |   |           ├── 000000.label
        |   |           ├── 000001.label  
        |   |           └── ...
        |   |	   ├── velodyne
        |   |	        ├── 000000.bin
        |   |           ├── 000001.bin
        |   |           └── ...
        │   ├── light/ 
        |   ├── moderate/
        ├── cross_sensor/
        ├── crosstalk/
        └── ...
        
```

### nuScenes-C
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
          ├──v1.0-trainval
          ├──beam_missing
          |     ├── heavy/
          |     |   ├── lidarseg/
          |     |   |   ├── v1.0-trainval
          |     |   |       ├── ....bin
          |     |   |       ├── ....bin 
          |     |   |       └── ...
          |     |   ├── sample/
          |     |       ├── LIDAR_TOP
          |     |           ├── ....pcd.bin
          |     |           ├── ....pcd.bin 
          |     |           └── ...
          |     ├── light/ 
          |     └── moderate/
          ├── cross_sensor/
          ├── crosstalk/
          └── ...
```


## Pretrained models

Pretrained model is trained on Synth4D. You can find the models [here](https://drive.google.com/file/d/1gT6KN1pYWj800qX54jAjWl5VGrHs8Owc/view?usp=sharing).

After downloading the pretrained models decompress them in ```D3CTTA/pretrained_models```.


## Adaptation to target

To adapt the source model to the target domain SemanticKITTI-C

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python adapt_online.py --config_file configs/adaptation/test_kitti.yaml 
```

To adapt the source model to the target domain NuScenes-C

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python adapt_online.py --config_file configs/adaptation/test_nusc.yaml 
```




## Thanks
We thank the open source projects [gipso-sfouda](https://github.com/saltoricristiano/gipso-sfouda), [Minkowski-Engine](https://github.com/NVIDIA/MinkowskiEngine).







