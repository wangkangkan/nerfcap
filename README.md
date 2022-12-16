# NerfCap: Human Performance Capture With Dynamic Neural Radiance Fields
### [Project Page](https://github.io/nerfcap) | [Video](http://www.cad.zju.edu.cn/home/gfzhang/papers/NerfCap/NerfCap_video.mp4) | [Paper](http://www.cad.zju.edu.cn/home/gfzhang/papers/NerfCap/NerfCap_TVCG_2022.pdf)
>Kangkan Wang, Sida Peng, Xiaowei Zhou, Jian Yang, Guofeng Zhang

## Installation
### Set up python environment
```
conda create -n nerfcap python=3.7
conda activate nerfcap

pip install torch==1.6.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Set up dataset
Download DeepCap dataset [here](https://gvv-assets.mpi-inf.mpg.de/).

## Run the code on DeepCap Dataset

### Test
1. Download the corresponding pretrained model and put it to
`$ROOT/data/trained_model/if_nerf/magdalena/latest.pth`

2. Test and visualization
    * Visualize all frames at test views
    `python run.py --type visualize --cfg_file configs/magdalena/magdalena.yaml exp_name magdalena`
    * Simultaneously extract mesh at each frame
    `python run.py --type visualize --cfg_file configs/magdalena/magdalena.yaml exp_name magdalena vis_mesh True`
3. The result are located at `$ROOT/data/result/if_nerf/magdalena`

### Train
1. Train
`python train_net.py --cfg_file configs/magdalena/magdalena.yaml exp_name magdalena resume False`
2. Tensorboard
`tensorboard --logdir data/record/if_nerf`

## Citation
If you find this code useful for your research, please use the following BibTeX entry.
```
@ARTICLE{NerfCap_TVCG22,
title={NerfCap: Human Performance Capture With Dynamic Neural Radiance Fields}, 
author={Wang, Kangkan and Peng, Sida and Zhou, Xiaowei and Yang, Jian and Zhang, Guofeng},  
journal={IEEE Transactions on Visualization and Computer Graphics},   
year={2022},  
pages={1-13},  
doi={10.1109/TVCG.2022.3202503}}
```
