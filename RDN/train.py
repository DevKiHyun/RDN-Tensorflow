import tensorflow as tf
import os
import time
import numpy as np
import cv2

from RDN.util import Time
from RDN.util import ImageBatch
from RDN.util import psnr

training_ratio = 1
main_data_path = '.'

# Source: https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
def im2double(image):
    info = np.iinfo(image.dtype) # Get the data type of the input image
    return image.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def get_train_set(scale=2):
    '''
    PATH of TRAIN SET(DIV2K-20 extracts)
    '''
    train_labels_path = '{}/data/train_DIV2K_label_{}x/*.npy'.format(main_data_path, scale)
    train_inputs_path = '{}/data/train_DIV2K_input_{}x/*.npy'.format(main_data_path, scale)
    '''
    TRAIN SET(N-Sampling) and shuffle
    '''
    train_inputs_batch = ImageBatch(train_inputs_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    train_labels_batch = ImageBatch(train_labels_path, training_ratio=training_ratio, on_sort=True, ext='npy')

    shuffle_indicese = list(range(train_labels_batch.N_TRAIN_DATA))
    np.random.shuffle(shuffle_indicese)
    train_inputs_batch.train_shuffle(shuffle_indicese)
    train_labels_batch.train_shuffle(shuffle_indicese)

    return train_inputs_batch, train_labels_batch

def get_test_set(batch_size, scale=2):
    '''
    PATH of TEST SET(SET5)
    '''
    test_labels_path = '{}/data/Set5/ground_truth/*.npy'.format(main_data_path)
    test_bicubic_inputs_path = '{}/data/Set5/blur_{}x/*.npy'.format(main_data_path, scale)
    test_rdn_inputs_path = '{}/data/Set5/low_rs_{}x/*.npy'.format(main_data_path, scale)
    '''
    TEST SET(SET5)
    '''
    test_labels_batch = ImageBatch(test_labels_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    test_bicubic_inputs_batch = ImageBatch(test_bicubic_inputs_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    test_rdn_inputs_batch = ImageBatch(test_rdn_inputs_path, training_ratio=training_ratio, on_sort=True, ext='npy')

    test_labels = test_labels_batch.next_batch(batch_size=batch_size)
    test_bicubic_inputs = test_bicubic_inputs_batch.next_batch(batch_size=batch_size)
    test_rdn_inputs = test_rdn_inputs_batch.next_batch(batch_size=batch_size)

    return test_labels, test_bicubic_inputs, test_rdn_inputs

def training(RDN, config):
    # Get training set and Test set
    train_inputs_batch, train_labels_batch = get_train_set(config.scale)
    test_labels, test_bicubic_inputs, test_rdn_inputs = get_test_set(batch_size=config.test_batch_size, scale=config.scale)

    '''
    Get bicubic average psnr of Test set
    '''
    avg_bicubic_psnr = 0
    for i in range(config.test_batch_size):
        label = test_labels[i]
        input = test_bicubic_inputs[i]
        avg_bicubic_psnr += psnr(label, input, peak=1)/5

    '''
    SETTING HYPERPARAMETER
    '''
    training_epoch = config.training_epoch
    batch_size = config.batch_size
    n_data = train_labels_batch.N_TRAIN_DATA
    total_batch = n_data // batch_size if n_data % batch_size == 0 else (n_data // batch_size) + 1
    total_iteration = training_epoch * total_batch
    n_iteration = 0

    # Build Network
    RDN.neuralnet()
    # Optimize Network
    RDN.optimize(config)
    # Summary of RDN Network
    RDN.summary()

    '''
    Train
    '''
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    writer = tf.summary.FileWriter('./model/rdn_result', sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print("Total the number of Data : " + str(n_data))
    print("Total Step per 1 Epoch: {}".format(total_batch))
    print("The number of Iteration: {}".format(total_iteration))

    for epoch in range(training_epoch):
        avg_cost = 0
        avg_rdn_psnr = 0
        test_psnr = [0 for i in range(config.test_batch_size)]
        for i in range(total_batch):
            start = time.time()

            batch_y = train_labels_batch.next_batch(batch_size, num_thread=8)
            batch_x = train_inputs_batch.next_batch(batch_size=batch_size, num_thread=8)

            summaries, _cost, _ = sess.run([RDN.summaries, RDN.cost, RDN.optimizer],
                                           feed_dict={RDN.X: batch_x, RDN.Y: batch_y})
            writer.add_summary(summaries, i)

            avg_cost += _cost / total_batch
            end = time.time()

            if epoch % 2 ==0 and i == 20:
                Time.require_time(start, end, count= total_iteration - n_iteration)

            n_iteration += 1

        if epoch % 2 == 0:
            '''
           Evaluate RDN performance (RGB Channel average psnr)
           '''
            for index in range(config.test_batch_size):
                label = test_labels[index].copy()
                input = test_rdn_inputs[index].copy()
                input = np.expand_dims(input, axis=0)

                result_rdn = sess.run(RDN.output, feed_dict={RDN.X: input})
                result_rdn = np.squeeze(result_rdn, axis=0)
                result_rdn = np.clip(result_rdn, 0, 1)

                _psnr = psnr(label, result_rdn, peak=1)
                test_psnr[index] = _psnr
                avg_rdn_psnr += _psnr/5

            print("=============================================")
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost),
                  '\nRGB AVG PSNR:: Bicubic: {:.9f} || RDN: {:.9f}'.format(avg_bicubic_psnr, avg_rdn_psnr))
            print("Evaluate RDN performance")
            for j in range(config.test_batch_size):
                print("Test image {} psnr: {}".format(j, test_psnr[j]))
            print("=============================================")

        shuffle_indicese = list(range(train_labels_batch.N_TRAIN_DATA))
        np.random.shuffle(shuffle_indicese)
        train_inputs_batch.train_shuffle(shuffle_indicese)
        train_labels_batch.train_shuffle(shuffle_indicese)

    print("학습 완료!")
    save_path = '{}/model/RDN.ckpt'.format(os.path.abspath('.'))
    saver.save(sess, save_path)
    print("세이브 완료")
