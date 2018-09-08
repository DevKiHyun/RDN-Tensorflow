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

![Alt Text](https://github.com/DevKiHyun/RDN-Tensorflow/blob/master/result/Urban100-1.gif)

## Reference

["The Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/pdf/1609.05158.pdf).
