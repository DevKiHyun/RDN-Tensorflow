# RDN-Tensorflow (2018/09/04)

## Introduction
We implement a tensorflow model for ["Residual Dense Network for Image Super-Resolution", CVPR 2018](https://arxiv.org/pdf/1802.08797.pdf).
- We use DIV2K dataset as training dataset.

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
- demo.py : Test the RDN model and show result images and psnr.
- util.py : Utility functions for this project.
- log.txt : The log of training process.
- model : The save files of the trained RDN.

## How to use
### Pre-processing

#### you should put the images of the DIV2K dataset into the 'DIV2K_train_HR' directory in data directory.
#### Input images(Low resolution) should be 48x48 size, so sub-images(High resolution) should be a specific multiple of the input image size. 
#### ex) Input images: 48x48 / [2x Scale] Sub_images : 96x96 [4x Scale] Sub_images : 196x196

##### Step 1
```shell
# Sampling N images in 'DIV2K_train_HR' directory
python sampling.py

# default args: n_extract = 30
# you can change args : n_extract = 20
python sampling.py --n_extract 20
```
##### Step 2
##### you should execute aug_train.m and aug_test.m in 'data' directory
##### Recommend 'Octave' platform to execute matlab code '.m' 

##### Step 3
```shell
# finally, you should execute preprocess.py
python preprocess.py
```

### Training
```shell
python main.py

# default args: training_epoch = 200, scale = 2, n_global_layers = 16, n_local_layers = 6 
# you can change args: training_epoch = 80, scale = 3, n_global_layers = 12, n_local_layers = 4
python main.py --training_epoch 80 --scale 3 --n_global_layers 12 --n_local_layer 4
```

### Test
```shell
python demo.py

# default args: image_index = 1, scale = 2, coordinate = [50,50], interval = 30 
# you can change args: image_index = 13, scale = 4, coorindate [100,100], interval = 50

python demo.py --image_index 13 --scale 4 --coordinate [100,100] --interval 50
```

## Result

##### Results on Urban 100/2 (visual)

![Alt Text](https://github.com/DevKiHyun/RDN-Tensorflow/blob/master/RDN/result/Urban100-1.gif)

## Reference

["The Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/pdf/1609.05158.pdf).
