# RGTNet

## Introduction

Implementation for paper: [Residual grounding transformer network for terrain recognition on the lunar surface](https://doi.org/10.1364/AO.428232 ).

## Requirements

- Packages

  The code was tested with Anaconda and Python 3.6.9. The Anaconda environment is:

  - numpy = 1.18.1
  - pillow = 7.1.2
  - tqdm = 4.46.1
  - matplotlib = 3.1.1
  - scikit-image = 0.16.2
  - opencv-python = 4.2.0.34
  - tensorboard = 2.2.2
  - tensorboardX = 2.0
- pytorch = 1.2.0
  
  - torchvision = 0.4.0a0
- cudatoolkit = 10.0.130
  - cudnn = 7.6.0
  

Install dependencies:

  - For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.
  - For custom dependencies:
  
    ```bash
    pip install matplotlib pillow tensorboardX tqdm
    ```

- 4 GPUs and a single GPU is also acceptable.

## Data Preparation

0. Put data in `./dataset` folder and Configure your dataset path in `mypath.py`.

   Make sure to put the data files as the following structure:

   ```
   ce3tr/train
   ├── image
   |   ├── 5-001.png
   │   ├── 5-002.png
   │   ├── 5-003.png
   │   ├── 5-004.png
   │   ├── ...
   |
   ├── mask
   |   ├── 5-001.png
   │   ├── 5-002.png
   │   ├── 5-003.png
   │   ├── 5-004.png
   |   ├── ...
   |   
   └── index
       ├── train.txt
       ├── val.txt
       ├── test.txt
       ├── all.txt
       ├── ...
   ```

   If you want to get the full **ce3tr** dataset, please contact '[qiulinwei@buaa.edu.cn](mailto:qiulinwei@buaa.edu.cn)'.

1. Get LBP feature maps.

   ```
   cd dataset/ce3tr/train/
   python get_LBP.py
   ```

   Then the LBP feature maps with `radius=8` are produced in folder `image1`.

   Change the settings in `get_LBP.py`:

   ```
   radius = 16
   to_path = './image2/'  # LBP output path
   ```

   Then the LBP feature maps with `radius=16` are produced in folder `image2`.

2. For other datasets like pascal, coco, cityscapes, it is the same with [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception).

## Model Training

Follow steps below to train your model:

1. Input arguments: (see full input arguments via python train.py --help):

   ```
   usage: train.py [-h] [--test]
                   [--backbone {resnet101,resnet50,xception,drn,mobilenet}]
                   [--out-stride OUT_STRIDE]
                   [--dataset {pascal,coco,cityscapes,ce3tr}] [--use-sbd]
                   [--workers N] [--base-size BASE_SIZE] [--crop-size CROP_SIZE]
                   [--sync-bn SYNC_BN] [--freeze-bn FREEZE_BN]
                   [--loss-type {ce,focal,dice,iou}] [--epochs N]
                   [--start_epoch N] [--batch-size N] [--test-batch-size N]
                   [--use-balanced-weights] [--lr LR]
                   [--lr-scheduler {poly,step,cos}] [--momentum M]
                   [--weight-decay M] [--nesterov] [--no-cuda]
                   [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                   [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                   [--no-val]
   
   ```

2. To train RGTNet using ce3tr dataset and ResNet101 as backbone with multi-GPUs:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet101 --lr 0.01 --epochs 500 --batch-size 4 --gpu-ids 0,1,2,3 --checkname ce3tr-resnet --dataset ce3tr --loss-type ce
   ```

3. To train RGTNet using ce3tr dataset and ResNet101 as backbone with a single GPU:

   ```bash
   python train.py --backbone resnet101 --lr 0.01 --epochs 500 --batch-size 2 --gpu-ids 0 --checkname ce3tr-resnet --dataset ce3tr --loss-type ce
   ```



## Model Testing

Run the testing script.

```bash
python test.py --dataset ce3tr --backbone {backbone/you/want/to/test} --model {path/to/your/checkpoint} --save_path {path/to/the/inference/result}
```

The meaning of some arguments:

- --no-resize: Input the images with original size.
- --no-blend: Not output the visualization results.



## Citation

If our work is useful for your research, please consider citing:

```tex
@article{qiu2021rgtnet,
  author = {Linwei Qiu, Haichao Li, ZhiLi and Cheng Wang},
  journal = {Appl. Opt.},
  number = {19},
  publisher = {OSA},
  title = {Residual grounding transformer network for terrain recognition on the lunar surface},
  volume = {60},
  month = {July},
  year = {2021},
  doi = {10.1364/AO.428232},
}
```

## Questions

Please contact '[qiulinwei@buaa.edu.cn](mailto:qiulinwei@buaa.edu.cn)'

## Acknowledgement

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[Feature Pyramid Transformer](https://github.com/dongzhang89/FPT)

## License

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/zh_CN)