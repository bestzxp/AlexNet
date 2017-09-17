import tensorflow as tf
import numpy as np
import cv2


class AlexNet(object):

    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        self.x = x
        self.num_classes = num_classes
        self.skip_layer = skip_layer
        self.keep_prob = keep_prob

        if weights_path == 'DEFAULT':
            self.weights_path = 'bvlc_alexnet.npy'
        else:
            self.weights_path = weights_path

        self.create()

    def create(self):

        '''
        创建网络框架结构
        :return:
        '''
        conv1 = self.conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = self.lrn(conv1, 2, 2e-5, 0.75, name='norm1')
        pool1 = self.max_pool(norm1, 3, 3, 2, 2, name='pool1', padding='VALID')

        conv2 = self.conv(pool1, 5, 5, 256, 1, 1, name='conv2', padding='VALID', groups=2)
        norm2 = self.lrn(conv2, 2, 2e-5, 0.75, name='norm2')
        pool2 = self.max_pool(norm2, 3, 3, 2, 2, name='pool2', padding='VALID')

        conv3 = self.conv(pool2, 3, 3, 384, 1, 1, name='conv3', padding='VALID')

        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, name='conv4', padding='VALID', groups=2)

        conv5 = self.conv(conv4, 3, 3, 256, 1, 1, name='conv5', padding='VALID')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, name='pool5', padding='VALID')

        flatten = tf.reshape((pool5, [-1, 6 * 6 * 256]))
        fc6 = self.fc(flatten, 6 * 6 * 256, 4096, name='fc6')
        dropout6 =self.dropout(fc6, self.keep_prob)

        fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.keep_prob)

        self.fc8 = self.fc(dropout7, 4096, self.num_classes, relu=False, name='fc8')


    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        input_channels = int(x.shape[-1])

        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_height, filter_width,
                                                        input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

            if groups == 1:
                conv = convolve(weights)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

                conv = tf.concat(axis=3, values=output_groups)

        biases = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        relu = tf.nn.relu(biases, name=scope.name)
        return relu

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    def fc(self, x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            return tf.nn.relu(act)
        else:
            return act
    def lodel_initial_weights(self, session):

        weights_dict = np.load(self.weights_path, encoding='bytes').items()

        for op_name in weights_dict:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))
