# RDN-Tensorflow (2018/09/04)

## Introduction
I implement a tensorflow model for ["Residual Dense Network for Image Super-Resolution", CVPR 2018](https://arxiv.org/pdf/1802.08797.pdf).
- I use DIV2K dataset as training dataset.

## Environment
- Ubuntu 16.04
- Python 3.5

## Depenency
- Numpy
- Opencv2
- matplotlib

## Files
- main.py : Execute train.py and pass the default value.
- vdsr.py : RDN model definition.
- train.py : Train the RDN model and represent the test set performance.
- test.py : Test the RDN model and show result images and psnr.
- demo.py : Upscale a input image by RDN model.
- util.py : Utility functions for this project.
- log.txt : The log of training process.
- model : The save files of the trained RDN.

## How to use
#### You can download a pre-processed training set [here](https://drive.google.com/file/d/1oqAlnACfGO8wkhqHJSuSWAnuIxw78POp/view?usp=sharing), then you don't need to follow pre-processing operation.(It is a training set for the 2x scale. I will soon also provide training sets for other scales.)
#### A pre-processed training set should be in the 'data' directory
### Pre-processing

#### You should put the images of the DIV2K dataset into the 'DIV2K_train_HR' directory in the 'data' directory.
#### Input images(Low resolution) should be 48x48 size, so sub-images(High resolution) should be a specific multiple of the input image size. 
#### ex) Input images: 48x48 / [2x Scale] Sub_images : 96x96 [4x Scale] Sub_images : 196x196

#### Step 1
##### [Recommend] 2x scale : 25 images sampling, 3x scale : 35 images sampling, 4x scale: 50 images sampling 
```shell
# Sampling N images in 'DIV2K_train_HR' directory
python sampling.py

# Default args: n_extract = 25 for 2x scale
# If 3x scale : n_extract = 35
python sampling.py --n_extract 35
# If 4x scale : n_extract = 50
python sampling.py --n-extract 50
```
#### Step 2
##### You should execute aug_train_'N'x.m and aug_test.m in 'data' directory. If you are training model for 2x scale, you must execute aug_train_2x.m
##### Recommend 'Octave' platform to execute matlab code '.m' 

#### Step 3
```shell
# Finally, you should execute preprocess.py
python preprocess.py
```
### If you do not want to train, skip the training step and unzip [model.zip](https://github.com/DevKiHyun/RDN-Tensorflow/tree/master/RDN/model) into the model directory.
### Training
```shell
python main.py

# Default args: training_epoch = 200, scale = 2, n_global_layers = 16, n_local_layers = 6 
# You can change args: training_epoch = 80, scale = 3, n_global_layers = 12, n_local_layers = 4
python main.py --training_epoch 80 --scale 3 --n_global_layers 12 --n_local_layer 4
```

### Test
```shell
python test.py

# Default args: image_index = 1, scale = 2, coordinate = [50,50], interval = 30 
# You can change args: image_index = 13, scale = 4, coorindate [100,100], interval = 50

python test.py --image_index 13 --scale 4 --coordinate [100,100] --interval 50
```
### Demo
```shell
python demo.py

# Default args: scale = 2
# You can change argg : scale = 4

python demo.py --scale 4
```

## Result

##### Results on Urban 100/2 (visual)

![Alt Text](https://github.com/DevKiHyun/RDN-Tensorflow/blob/master/RDN/result/Urban100-1.gif)

## Reference

["The Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/pdf/1609.05158.pdf).
