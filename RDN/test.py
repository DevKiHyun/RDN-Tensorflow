import argparse
import tensorflow as tf
import numpy as np
import cv2
import os

import RDN.rdn as rdn
from RDN.util import ImageBatch
from RDN.util import display
from RDN.util import psnr

# Source: https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
def im2double(image):
    info = np.iinfo(image.dtype) # Get the data type of the input image
    return image.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def modcrop(image, scale):
    height, width, _ = image.shape
    height = height - (height % scale)
    width = width - (width % scale)
    image = image[0:height, 0:width, :]

    return image

def bicubic_sr(input, scale):
    height, width, n_channel = input.shape

    bicubic_output =  np.clip(cv2.resize(
                        cv2.resize(input.copy(), None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC),
                         (width, height), interpolation=cv2.INTER_CUBIC), 0, 1)*255

    return bicubic_output

def RDN_sr(sess, RDN, input, scale):
    low_rs = np.clip(cv2.resize(input.copy(), None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC), 0, 1)
    RDN_output = sess.run(RDN.output, feed_dict={RDN.X:np.expand_dims(low_rs, axis=0)})[0]
    RDN_output = np.clip(RDN_output*255, 0, 255)

    return RDN_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_global_layers', type=int, default=16, help='-')
    parser.add_argument('--n_local_layers', type=int, default=6, help='-')
    parser.add_argument('--n_channel', type=int, default=3, help='-')
    parser.add_argument('--batch_size', type=int, default=100, help='-')
    parser.add_argument('--image_index', type=int, default=1, help='-')
    parser.add_argument('--scale', type=int, default=2, help='-')
    parser.add_argument('--coordinate', type=int, default=[50, 50], help='-')
    parser.add_argument('--interval', type=int, default=30, help='-')
    args, unknown = parser.parse_known_args()

    test_y_images_path = './data/Urban100/HR/*.png'
    result_save_path = './result'
    if not os.path.exists(result_save_path): os.makedirs(result_save_path)

    labels_set = ImageBatch(test_y_images_path, training_ratio=1, on_sort=True, ext='png')
    labels = labels_set.next_batch(batch_size=args.batch_size)

    RDN = rdn.RDN(args)
    RDN.neuralnet()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    saver = tf.train.Saver()
    saver.restore(sess, './model/RDN.ckpt')

    index = args.image_index
    scale = args.scale
    x_start = args.coordinate[0]
    y_start = args.coordinate[1]
    interval = args.interval

    label = modcrop(labels[index], scale=scale)
    input = im2double(label.copy().astype(np.uint8))

    bicubic_output = bicubic_sr(input.copy(), scale=scale)
    RDN_output = RDN_sr(sess, RDN, input.copy(), scale=scale)

    low_rs = np.clip(cv2.resize(input.copy(), None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC), 0,
                     1)*255

    cv2.imwrite('{}/{}.png'.format(result_save_path, 'low'), low_rs)
    cv2.imwrite('{}/{}.png'.format(result_save_path, 'original'), label)
    cv2.imwrite('{}/{}.png'.format(result_save_path, 'bicubic'), bicubic_output)
    cv2.imwrite('{}/{}.png'.format(result_save_path, 'RDN'), RDN_output)

    print("Bicubic PSNR: ", psnr(label, bicubic_output))
    print("RDN PSNR: ", psnr(label, RDN_output))

    #input_list = [input, input[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #bicubic_list = [bicubic_output, bicubic_output[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #VDSR_list = [VDSR_output, VDSR_output[x_start:x_start+interval,  y_start:y_start+interval, :]]

    original_list = np.array([label, bicubic_output, RDN_output])

    #zoom_list = np.array(original_list[:])
    display_list = np.array([original_list])
    display(display_list)

    #display_list = np.array([original_list, zoom_list)
    #display(display_list,  figsize = (5,5), axis_off=True, size_equal=True, gridspec=(0,0), zoom_coordinate=(150, 190, 100,260))