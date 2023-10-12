# EvalProtoPNet

## Introduction

This is the official implementation of paper **"Evaluation and Improvement of Interpretability for Self-Explainable Part-Prototype Networks"**.

## Required Packages

Our required python packages are listed as below:

```
pytorch==1.12.1+cu113
Augmentor==0.2.9
torchvision==0.13.1+cu113
pillow==8.4.0 (9.3.0 for train)
timm==0.5.4
opencv-python==4.6.0.66
tensorboard==2.9.1
scipy==1.8.1
pandas==1.4.3
matplotlib==3.5.2
scikit-learn==1.1.1
numpy==1.22.0
tqdm==4.66.1
```

## Dataset Preparation

* Download the dataset (CUB_200_2011.tgz) from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).
* Unpack CUB_200_2011.tgz to the `datasets/` directory in this project (the path of CUB-200-2011 dataset will be `datasets/CUB_200_2011/`).
* Run `python util/crop_cub_data.py` to split the cropped images into training and test sets. The cropped training images will be in the directory `datasets/cub200_cropped/train_cropped/`, and the cropped test images will be in the directory `datasets/cub200_cropped/test_cropped/`.
* Run `python util/img_aug.py --data_path /path/to/source_codes/datasets/cub200_cropped` to augment the training set. Note that `/path/to/source_codes/datasets/cub200_cropped` should be an absolute path. This will create an augmented training set in the following directory: `datasets/cub200_cropped/train_cropped_augmented/`.

## Model Weights

Most model weights will be downloaded automatically, except the ResNet50 backbone on iNaturalist2017. It (`BBN.iNaturalist2017.res50.90epoch.best_model.pth`) can be downloaded from [here](https://drive.google.com/drive/folders/1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU) and put into the folder `pretrained_models`.

## Training Instructions

Use `scripts/train.sh` for training:

```
sh scripts/train.sh $model $num_gpus
```

Here, `$model` is the name of backbone chosen from `resnet34, resnet152, vgg19, densenet121, densenet161`, `$num_gpus` is the number of GPUs for training (2 GPUs is recommended).

For example, the instruction for training a ResNet34 model with 2 GPUs is as below:

```
sh scripts/train.sh resnet34 2
```

Note that when running two scripts at the same time, the variable `use_port` in `scripts/train.sh` should be different for them.

## Evaluate Consistency Score

The instruction for evaluating the consistency score of a ResNet34 model with checkpoint path `$ckpt_path`:

```
python eval_consistency.py \
--base_architecture resnet34 \
--resume $ckpt_path
```

## Evaluate Stability Score

The instruction for evaluating the stability score of a ResNet34 model with checkpoint path `$ckpt_path`:

```
python eval_stability.py \
--base_architecture resnet34 \
--resume $ckpt_path
```

## Visualization

The instruction for visualizing a ResNet34 model with checkpoint path `$ckpt_path` for the images in category `$class`:

```
python local_analysis_vis.py \
--base_architecture resnet34 \
--resume $ckpt_path \
--imgclass $class
```

For example, the instruction for evaluating a ResNet34 model with checkpoint path `output_cosine/CUB2011/resnet34/1028-1e-4-adam-12-train/checkpoints/save_model.pth` for the images in category `15`:

```
python local_analysis_vis.py \
--base_architecture resnet34 \
--resume output_cosine/CUB2011/resnet34/1028-1e-4-adam-12-train/checkpoints/save_model.pth \
--imgclass 15
```

The results will be saved in `output_view` directory.
