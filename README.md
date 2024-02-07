# savid: Spectravista Aesthetic Vision Integration for Robust and Discerning 3D Object Detection in Challenging Environments (AAAI24)
<p align="center"> <img src='docs/savid_framework.png' align="center" height="300px"> </p>

This is the official implementation of [**savid: Spectravista Aesthetic Vision Integration for Robust and Discerning 3D Object Detection in Challenging Environments**], built on [`Deep Fusion`](https://github.com/tensorflow/lingvo) and [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) 


### Installation
1.  Prepare for the running environment. 

    You can use the docker image provided by [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). Our experiments are based on the
    docker provided by Voxel-R-CNN and we use 8 TITAN V GPUs to train our savid.

2. Prepare for the data.

    Please prepare dataset as [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet).  
    
    To generate depth_pseudo_rgbseguv_twise by yourself with depth_dense_twise as follows:

    ```
    cd savid
    python depth_to_lidar.py
    ```
    
    If you want to generate dense depth maps by yourself, it is recommended to use [`TWISE`](https://github.com/imransai/TWISE). The dense depth maps we provide are generated by TWISE. Anyway, you should have your dataset as follows:

    ```
    savid
    ├── data
    │   ├── argoverse_savid_seguv_twise
    │   │   │── ImageSets
    │   │   │── training
    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & depth_dense_twise & depth_pseudo_rgbseguv_twise
    │   │   │── testing
    │   │   │   ├──calib & velodyne & image_2 & depth_dense_twise & depth_pseudo_rgbseguv_twise
    ├── pcdet
    ├── tools
    ```
    Each pseudo point in depth_pseudo_rgbseguv_twise has 9 attributes (x, y, z, r, g, b, seg, u, v). It should be noted that we do not use the seg attribute, because the image segmentation results cannot bring improvement to savid in our experiments.

3. Setup.

    ```
    cd savid
    python setup.py develop
    cd pcdet/ops/iou3d/cuda_op
    python setup.py develop
    cd ../../../..
    ```

### Getting Started
1. Training.

    ```
    cd savid/tools
    scripts/dist_train.sh 8 --cfg_file cfgs/argoverse_models/savid.yaml --gpu_id 0,1,2,3,4,5,6,7
    ```

2. Evaluation.

    ```
    cd savid/tools
    scripts/dist_test.sh 8 --cfg_file cfgs/argoverse_models/savid.yaml  --gpu_id 0,1,2,3,4,5,6,7 --batch_size 28 \
    --ckpt ../output/argoverse_models/savid/default/ckpt/checkpoint_epoch_58.pth
    ```
    

### Weights will be provided once the paper is accepted.
### Thank you 🙏.