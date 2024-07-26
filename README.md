# Strategic Advancements in Semi-supervised Medical Segmentation with Emphasis on Local Accuracy and Label Precision
  
  Wen-Shan Hsu*, Kai-Lung Hua

</div>


## Table of Contents
- [Overview](#overview)
  - [Contributions](#contributions)
  - [ExperimentTags](#experiments-tags-todo-tree)
  - [BaselineModel](#baseline-model)
- [Results](#results)
  - [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Evaluation](#kitti-evaluation)
- [Training](#training)
  - [Dependency Installation](#dependency-installation)
  - [Start Training](#start-training)

## Overview
Medical image segmentation faces significant challenges due to the scarcity of annotated data and the abundance of unlabeled data. Our study utilizes a semi-supervised learning approach centered on an enhanced teacher-student model framework to harness this potential.

### Contributions
1. We propose a method that combines dynamic mask adjustment with bidirectional copy-paste, which not only reduces the gap between unlabeled and labeled data but also ensures that the model can accurately segment images even when parts of the image are occluded.
2. We integrate the Edge-Guided Module (EGM) into UNet and VNet to enhance image boundaries using edge signals, improving segmentation accuracy, especially with blurred boundaries.
3. We use a dual-teacher model strategy with a dynamic teacher reflecting the latest learning state and a static teacher providing stable supervision through periodic updates. This approach leverages their strengths to enhance pseudo-label accuracy and improve the student model's learning and performance.

### Code Changed
1. code/ACDC_labelnum3_train.py
2. code/ACDC_labelnum7_train.py
3. code/LA_labelnum4_train.py
4. code/LA_labelnum8_train.py
5. networks/net_factory_ega.py
6. networks/net_factory_la_ega.py
7. networks/unet_ega.py
8. networks/VNet_la_ega.py
9. pancreas/dataloaders_all.py
10. pancreas/pancreas_labelnum20_train.py
11. pancreas/pancreas_utils_all.py
12. pancreas/Vnet_ega_all.py


### Baseline Model:
Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation [paper link](https://arxiv.org/abs/2305.00673)

### Dataset:
You can download the ACDC dataset, LA dataset, pancreas_NIH dataset from this [link](https://drive.google.com/drive/folders/1ePzJ4OOgc4het369iFXlPTyFwLsguQGg?usp=sharing)

### Results
You can download the model weight from this [link](https://drive.google.com/drive/folders/1DMPWVQdXP1Zicieua1mj08eSBRmjoCL9?usp=sharing)

### Environment
```
MY DEVICES:
    OS:  Ubuntu 20.04
    GPU: Nvidia Geforce RTX 2080 x2
    PyTorch:
         CUDA 11.1
         PyTorch 1.8.1
         cudnn 8
    Python 3.8.8
```

### Data Preparation
- Download ACDC dataset and LA dataset, put them in my_all_code/, named data_spilt
- Download pancreas-NIH dataset, put them in my_all_code/code/pancreas, named data_lists



### Evaluation
* If you want to inference ACDC dataset with 3% labeled data, you need to enter ```python test_ACDC.py --labelnum 3```, and need to put ACDC_3_model.pth into my_all_code/code/model/BCP/ACDC_BCP_3_labeled/self_train, and renamed "unet_best_model.pth"

* If you want to inference ACDC dataset with 7% labeled data, you need to enter ```python test_ACDC.py --labelnum 7```, and need to put ACDC_7_model.pth into my_all_code/code/model/BCP/ACDC_BCP_7_labeled/self_train, and renamed "unet_best_model.pth"

* If you want to inference LA dataset with 4% labeled data, you need to enter ```python test_LA.py --labelnum 4```, and need to put LA_4_model.pth into my_all_code/code/model/BCP/LA_BCP_4_labeled/self_train, and renamed "VNet_best_model.pth"

* If you want to inference LA dataset with 8% labeled data, you need to enter ```python test_LA.py --labelnum 8```, and need to put LA_8_model.pth into my_all_code/code/model/BCP/LA_BCP_8_labeled/self_train, and renamed "VNet_best_model.pth"

* If you want to inference pancreas-NIH dataset, you need to enter ```python test_pancreas.py```, and need to put pancreas_20_model.pth into my_all_code/code/pancreas/result/cutmix/self_train/, and renamed "best_ema_20_self.pth"



## Training
#### dependency installation 
    pip install -r requirement.txt
    
#### start training
* If you want to train ACDC dataset with 5% labeled data, you need to enter ```python ACDC_labelnum3_train.py --run_mode pretrain_then_selftrain```
* If you want to train ACDC dataset with 10% labeled data, you need to enter ```python ACDC_labelnum7_train.py --run_mode pretrain_then_selftrain```
* If you want to train LA dataset with 5% labeled data, you need to enter ```python LA_labelnum4_train.py ```
* If you want to train LA dataset with 10% labeled data, you need to enter ```python LA_labelnum8_train.py ```
* If you want to train pancreas-NIH dataset with 20% labeled data, you need to enter ```python -m torch.distributed.launch pancreas_labelnum20_train.py ```

#### Tips:
* If you failed to training, maybe can enter ```apt-get install libx11-6```
