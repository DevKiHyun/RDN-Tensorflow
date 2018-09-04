import tensorflow as tf
import numpy as np

class RDN:
    def __init__(self, config):
        self.n_channel = config.n_channel
        self.weights = {}
        self.biases = {}
        self.X = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channel])
        self.Y = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channel])
        self.n_global_layers = config.n_global_layers
        self.n_local_layers = config.n_local_layers
        self.scale = config.scale #tf.Variable(2, dtype=tf.int32, trainable=False)
        self.images_mean = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.psnr = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

    def _conv2d_layer(self, inputs, filters_size, strides=[1, 1], add_bias=False, name=None,
                      padding="SAME", activation=None, stddev=None):

        filters = self._get_conv_filters(filters_size, name, stddev=stddev)
        strides = [1, *strides, 1]

        conv_layer = tf.nn.conv2d(inputs, filters, strides=strides, padding=padding, name=name + "_layer")

        if add_bias != False:
            conv_layer = tf.add(conv_layer, self._get_bias(filters_size[-1], name))
        if activation != None:
            conv_layer = activation(conv_layer)

        return conv_layer

    def _get_conv_filters(self, filters_size, name, stddev=None):
        name = name + "_weights"

        if stddev == None:
            initializer = tf.contrib.layers.xavier_initializer()
            conv_weights = tf.Variable(initializer(shape=filters_size), name=name)
        else:
            initializer = tf.random_normal
            conv_weights = tf.Variable(initializer(shape=filters_size, stddev=stddev), name=name)
        self.weights[name] = conv_weights

        return conv_weights

    def _get_bias(self, bias_size, name):
        name = name + "_bias"
        bias = tf.Variable(tf.zeros([bias_size]), name=name)
        self.biases[name] = bias

        return bias

    def _residual_block(self, inputs, filters_size, name=None, activation=None):
        after_channel = filters_size[-1]
        layer_list = []
        layer_list.append(inputs)

        local_conv = self._conv2d_layer(inputs, filters_size=filters_size, add_bias=True,
                                        name=name+"local_conv_0", activation=activation)
        layer_list.append(local_conv)
        for i in range(self.n_local_layers):
            local_conv = self._conv2d_layer(local_conv, filters_size=filters_size, add_bias=True,
                                        name=name+"local_conv_{}".format(i+1), activation=activation)
            layer_list.append(local_conv)

        local_fusion = tf.concat(layer_list, axis=3)
        local_after_channel = local_fusion.get_shape().as_list()[-1]
        block_output = self._conv2d_layer(local_fusion, filters_size=[1,1,local_after_channel,after_channel], name=name+"fusion")

        return block_output

    def _upscale(self, inputs, scale):
        # According to this paper [https://arxiv.org/pdf/1609.05158.pdf]
        conv = self._conv2d_layer(inputs, filters_size=[5,5,64,64], add_bias=True,
                                  name="upscale_{}_0".format(scale), activation=tf.nn.relu)
        conv = self._conv2d_layer(conv, filters_size=[3,3,64,32], add_bias=True,
                                  name="upscale_{}_1".format(scale), activation=tf.nn.relu)

        conv = self._conv2d_layer(conv, filters_size=[3,3,32,self.n_channel*np.power(scale,2)], add_bias=True,
                                  name="upscale_{}_2".format(scale))

        upscaled_conv = tf.depth_to_space(conv, scale)

        return upscaled_conv

    def neuralnet(self):
        global_layer_list = []

        conv_F_1 = self._conv2d_layer(self.X, filters_size=[3,3,self.n_channel, 64], add_bias=True,
                                  name="conv_0", activation=tf.nn.relu)
        conv = self._conv2d_layer(conv_F_1, filters_size=[3,3,64,64], add_bias=True,
                                  name="conv_1", activation=tf.nn.relu)
        '''
        Residual Dense Block
        '''
        for i in range(self.n_global_layers):
            conv = self._residual_block(conv, filters_size=[3,3,64,64],
                                        name="conv_residual_{}".format(i), activation=tf.nn.relu)
            global_layer_list.append(conv)

        '''
        Dense Feature Fusion
        '''
        global_fusion = tf.concat(global_layer_list, axis=3)
        global_after_channel = global_fusion.get_shape().as_list()[-1]
        conv = self._conv2d_layer(global_fusion, filters_size=[1,1,global_after_channel, 64], name="global_fusion")
        conv = self._conv2d_layer(conv, filters_size=[3,3,64,64], name="global_conv")
        conv = tf.add(conv, conv_F_1)

        '''
        Upscaling
        '''
        upscaled_conv = self._upscale(conv, self.scale)
        HR_conv = self._conv2d_layer(upscaled_conv, filters_size=[3, 3, self.n_channel, self.n_channel], add_bias=True,
                                     name="HR_conv")
        self.output = HR_conv


    def optimize(self, config):
        self.learning_rate = config.learning_rate
        self.beta_1 = config.beta_1
        self.beta_2 = config.beta_2
        self.epsilon = config.epsilon

        self.cost = tf.reduce_mean(tf.pow(self.Y - self.output, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1,
                                                beta2=self.beta_2, epsilon=self.epsilon).minimize(self.cost)

    def summary(self):
        '''
        for weight in list(self.weights.keys()):
            tf.summary.histogram(weight, self.weights[weight])
        for bias in list(self.biases.keys()):
            tf.summary.histogram(bias, self.biases[bias])
        '''

        #tf.summary.scalar('Loss', self.cost)
        tf.summary.scalar('Average test psnr', self.psnr)
        tf.summary.scalar('Learning rate', self.learning_rate)

        self.summaries = tf.summary.merge_all()