# AOT (Associating Objects with Transformers) in PyTorch


A modular reference PyTorch implementation of Associating Objects with Transformers for Video Object Segmentation (NIPS 2021). [[paper](https://arxiv.org/abs/2106.02638)]

![alt text](source/overview.png "Overview.")

![alt text](source/some_results.png "Some results.")

## Highlights
- **High performance:** up to **85.5%** (R50-AOTL) on YouTube-VOS 2018 and **82.1%** (SwinB-AOTL) on DAVIS-2017 Test-dev under standard settings. 
- **High efficiency:** up to **51fps** (AOTT) on DAVIS-2017 (480p) even with **10** objects and **41fps** on YouTube-VOS (1.3x480p). AOT can process multiple objects (less than a pre-defined number, 10 in default) as efficiently as processing a single object. This project also supports inferring any number of objects together within a video by automatic separation and aggregation.
- **Multi-GPU training and inference**
- **Mixed precision training and inference** 
- **Test-time augmentation:** multi-scale and flipping augmentations are supported.


## TODO
- [ ] Code documentation
- [ ] Demo tool
- [ ] Adding your own dataset
- [ ] Results with test-time augmentations in Model Zoo

## Requirements
   * Python3
   * pytorch >= 1.7.0 and torchvision
   * opencv-python
   * Pillow

Optional (for better efficiency):
   * Pytorch Correlation (recommend to install from [source](https://github.com/ClementPinard/Pytorch-Correlation-extension) instead of using `pip`)

## Demo
Coming

## Model Zoo and Results
Pre-trained models and corresponding results reproduced by this project can be found in [MODEL_ZOO.md](MODEL_ZOO.md).

## Getting Started
1. Prepare datasets:

    Please follow the below instruction to prepare datasets in each correspondding folder.
    * **Static** 
    
        `datasets/Static`: pre-training dataset with static images. A guidance can be found in [AFB-URR](https://github.com/xmlyqing00/AFB-URR).
    * **YouTube-VOS**

        A commonly-used large-scale VOS dataset.

        `datasets/YTB/2019`: version 2019, download [link](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz?usp=sharing). `train` is required for training. `valid` (6fps) and `valid_all_frames` (30fps, optional) are used for evaluation.

        `datasets/YTB/2018`: version 2018, download [link](https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f?usp=sharing). Only `valid` (6fps) and `valid_all_frames` (30fps, optional) are required for this project and used for evaluation.

    * **DAVIS**

        A commonly-used small-scale VOS dataset.

        `datasets/DAVIS`: [TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) (480p) contains both the training and validation split. [Test-Dev](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip) (480p) contains the Test-dev split. The [full-resolution version](https://davischallenge.org/davis2017/code.html) is also supported for training and evaluation but not required.


2. Prepare ImageNet pre-trained encoders

    Select and download below checkpoints into `pretrain_models`:

    - [MobileNet-V2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) (default encoder)
    - [MobileNet-V3](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth)
    - [ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth)
    - [ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth)
    - [ResNeSt-50](https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth)
    - [ResNeSt-101](https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth)
    - [Swin-Base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)

    The current default training configs are not optimized for encoders larger than ResNet-50. If you want to use larger encoders, we recommond to early stop the main-training stage at 80,000 iteration (100,000 in default) to avoid over-fitting on the seen classes of YouTube-VOS.



3. Training and Evaluation

    The [example script](train_eval.sh) will train AOTT with 2 stages using 4 GPUs and auto-mixed precision (`--amp`). The first stage is a pre-training stage using `Static` dataset, and the second stage is main-training stage, which uses both `YouTube-VOS 2019 train` and `DAVIS-2017 train` for training, resulting in a model can generalize to different domains (YouTube-VOS and DAVIS) and different frame rates (6fps, 24fps, and 30fps).

    Notably, you can use only the `YouTube-VOS 2019 train` split in the second stage by changing `pre_ytb_dav` to `pre_ytb`, which leads to better YouTube-VOS performance on unseen classes. Besides, if you don't want to do the first stage, you can start the training from stage `ytb`, but the performance will drop about 1~2% absolutely.

    After the training is finished (about 0.6 day for each stage with 4 Tesla V100 GPUs), the [example script](train_eval.sh) will evaluate the model on YouTube-VOS and DAVIS, and the results will be packed into Zip files. For calculating scores, please use offical YouTube-VOS servers ([2018 server](https://competitions.codalab.org/competitions/19544) and [2019 server](https://competitions.codalab.org/competitions/20127)) and offical [DAVIS toolkit](https://github.com/davisvideochallenge/davis2017-evaluation).


## Adding your own dataset
Coming

## Troubleshooting
Waiting

## Citations
Please consider citing the related paper(s) in your publications if it helps your research.
```
@inproceedings{yang2021aot,
  title={Associating Objects with Transformers for Video Object Segmentation},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## License
This project is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for additional details.
