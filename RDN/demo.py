import argparse
import tensorflow as tf
import numpy as np
import cv2
import os
import glob

import RDN.rdn as rdn

# Source: https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
def im2double(image):
    info = np.iinfo(image.dtype) # Get the data type of the input image
    return image.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def bicubic_sr(input, scale):
    bicubic_output = np.clip(cv2.resize(input.copy(), None, fx=1.0*scale, fy=1.0*scale, interpolation=cv2.INTER_CUBIC),0,1) * 255
    bicubic_output = bicubic_output

    return bicubic_output

def RDN_sr(sess, RDN, input):
    RDN_output = sess.run(RDN.output, feed_dict={RDN.X:np.expand_dims(input, axis=0)})[0]
    RDN_output = np.clip(RDN_output*255, 0, 255)
    RDN_output = RDN_output

    return RDN_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_global_layers', type=int, default=16, help='-')
    parser.add_argument('--n_local_layers', type=int, default=6, help='-')
    parser.add_argument('--n_channel', type=int, default=3, help='-')
    parser.add_argument('--scale', type=int, default=2, help='-')
    args, unknown = parser.parse_known_args()

    scale = args.scale

    result_save_path = './demo_result'
    if not os.path.exists(result_save_path): os.makedirs(result_save_path)

    print("Please enter the path of the image's directory. Ex) '/data/test_images' ")
    path = input("Enter the directory path: ")
    print("Please enter the name of images. This version 'only' read '.png', 'jpg' extention and 'RGB' image! Ex) '0.png'")
    print("If you want to read all images in the directory, enter '*.png' or '*.jpg'.")
    name = input("Enter the image name: ")

    image_path = path + '/' + name
    images_list = glob.glob(image_path)

    if not images_list:
        print("Path is empty. Check that the directory or image name is correct.")
    else:
        RDN = rdn.RDN(args)
        RDN.neuralnet()

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, './model/RDN.ckpt')

        for image in images_list:
            filename = os.path.basename(image)[:-4]

            label = cv2.imread(image).astype(np.uint8)
            height, width =  label.shape[:2]

            if len(label.shape) != 3:
                print("{}.png is not RGB 3 channel image".format(filename))
                break
            elif height*width > 750*750:
                print("{}.png' size is large. Recommend a size smaller than 750x750.".format(filename))
            else:
                input = im2double(label.copy())

                RDN_upscale = RDN_sr(sess, RDN, input.copy())
                bicubic_upscale = bicubic_sr(input.copy(), scale=scale)

                cv2.imwrite('{}/{}.png'.format(result_save_path, '{}_bicubic_{}x'.format(filename, scale)), bicubic_upscale)
                cv2.imwrite('{}/{}.png'.format(result_save_path, '{}_RDN_{}x'.format(filename, scale)), RDN_upscale)

