import glob
import numpy as np
import scipy.io
import os

train_input_2x_path = './data/train_DIV2K_input_2x/{}'
train_input_3x_path = './data/train_DIV2K_input_3x/{}'
train_input_4x_path = './data/train_DIV2K_input_4x/{}'
train_label_2x_path = './data/train_DIV2K_label_2x/{}'
train_label_3x_path = './data/train_DIV2K_label_2x/{}'
train_label_4x_path = './data/train_DIV2K_label_2x/{}'

train_path = [train_input_2x_path, train_input_3x_path, train_input_4x_path, train_label_2x_path, train_label_3x_path, train_label_4x_path]
test_set = ['data/Set5']
test_path = []
for elem in test_set:
    y_ch_path = './' + elem + '/ground_truth/{}'
    test_path.append(y_ch_path)
    y_ch_bicubic_2x_path = './' + elem + '/blur_2x/{}'
    test_path.append(y_ch_bicubic_2x_path)
    y_ch_bicubic_3x_path = './' + elem + '/blur_3x/{}'
    test_path.append(y_ch_bicubic_3x_path)
    y_ch_bicubic_4x_path = './' + elem + '/blur_4x/{}'
    test_path.append(y_ch_bicubic_4x_path)
    y_ch_rdn_2x_path = './' + elem + '/low_rs_2x/{}'
    test_path.append(y_ch_rdn_2x_path)
    y_ch_rdn_3x_path = './' + elem + '/low_rs_3x/{}'
    test_path.append(y_ch_rdn_3x_path)
    y_ch_rdn_4x_path = './' + elem + '/low_rs_4x/{}'
    test_path.append(y_ch_rdn_4x_path)

for path in train_path:
    list = glob.glob(path.format('*.mat'))
    for file in list:
        print(file, ' --> npy')
        filename = os.path.basename(file)[:-4]
        mat = scipy.io.loadmat(file)['patch']
        np.save('{}.npy'.format(path.format(filename)), mat)
        os.remove(file)

for path in test_path:
    list = glob.glob(path.format('*.mat'))
    for file in list:
        print(file, '--> npy')
        filename = os.path.basename(file)[:-4]
        mat = scipy.io.loadmat(file)['img']
        np.save('{}.npy'.format(path.format(filename)), mat)
        os.remove(file)
