#-*-coding utf-8-*-
import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
'''
Model : VesselDNN

VesselDNN은 Fundus image 내에서 Pixel-wise로 혈관인지 아닌지를 분류한다.
이미지 전체를 Unit Patch(27X27)로 나눈 후, 이것을 input으로 하여 구분한다.

Conv Layer가 4층으로 이루어져 있고(64,64,128,128)
FC Layer가 2층으로 이루어져 있는 (512,512) 단순한 구조이다.

성능향상을 위해선, image_utils에서 이미지 전처리를 크게
CLAHE 알고리즘과 GCN(global Contrast Normalization)을 통해 높혔다.

장점은 적은 이미지 수를 바탕으로도, 충분히 데이터를 확보할 수 있다는 점이고,
단점은 장 당 처리 시간이 다른 모델에 비해 길다는 점이다.

이 코드에서는 graph def를 캡슐화 하였으므로, 학습을 시키기 위해서는

model = VesselDNN()
with tf.Session(graph=model.graph) as sess:
    sess.run(init)
    ~~~~~~~
로 진행해야 한다.

Reference : Segmenting Retinal Blood vessels with Deep Neural Networks
'''
class VesselDNN(object):
    in_node = None
    label = None
    is_training = None
    keep_prob = None

    logits = None  # output tensor
    hypothesis = None # Softmax value
    predictor = None # argmax value

    graph = None  # graph (tensorflow metadata)
    cost = None
    # indicator of performance
    accuracy = None
    iou = None

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('VesselDNN') # logging
        self.logger.info("VesselDNN initializing start!")
        # the speicification of image and label shape
        self.height = kwargs.get('height', 27)  # the height size of image
        self.width = kwargs.get('width', 27)  # the width size of image
        self.channels = kwargs.get('channels', 3)  # the channel size of image
        self.n_class = kwargs.get('n_class', 2)  # the number of output labels

        # the size of convolutional filter
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.pool_size = kwargs.get('pool_size', 2)  # max-pooling size
        self.l2_scale = kwargs.get('l2_scale', 5e-4)  # l2 regularizer value

        self.logger.info("[height,width,channels,n_class]:[{},{},{},{}]"\
            .format(self.height,self.width,self.channels,self.n_class))

        with tf.Graph().as_default() as graph:
            # input tensor
            self.in_node = tf.placeholder(
                tf.float32, [None, self.height, self.width, self.channels], name="input")
            self.label = tf.placeholder(
                tf.float32, [None, self.n_class], name="label")
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # Convolution filter initializer
            xavier_init = tf.contrib.layers.xavier_initializer_conv2d()
            # l2 regularizer tensor
            if self.l2_scale is not None:
                self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_scale)
            else:
                self.regularizer = None

            conv1 = tf.layers.conv2d(self.in_node, filters=64, kernel_size=3,strides=(1,1),
                                        kernel_initializer=xavier_init,padding='same',
                                        activation=tf.nn.relu, name='conv1')
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3,strides=(1,1),
                                    kernel_initializer=xavier_init,padding='same',
                                    activation=tf.nn.relu, name='conv2')
            conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=3,strides=(1,1),
                                    kernel_initializer=xavier_init,padding='same',
                                    activation=tf.nn.relu, name='conv3')
            conv4 = tf.layers.conv2d(conv3, filters=128, kernel_size=3,strides=(1,1),
                                    kernel_initializer=xavier_init,padding='same',
                                    activation=tf.nn.relu, name='conv4')
            flatten = tf.contrib.layers.flatten(conv4)
            fc1 = self._fc_dropout_relu(flatten, 'fc1', 512)
            fc2 = self._fc_dropout_relu(fc1, 'fc2', 512)

            with tf.variable_scope("out"):
                self.logits = tf.layers.dense(fc2, 2, activation=None, name='logits')
                self.hypothesis = tf.nn.softmax(self.logits, name='softmax')
                self.predictor = tf.argmax(self.hypothesis,axis=1,name="predict")

            self.logger.info("[output] output shape : {}".format(self.logits.shape))
            self._get_cost() # cost tensor
            self._get_acc() # accuracy tensor
            self.graph = graph

    def _fc_dropout_relu(self, in_node, scope, units):
        with tf.variable_scope(scope):
            h1 = tf.layers.dense(in_node, units=units, activation=None,
                                  kernel_regularizer=self.regularizer, name='fc')
            h2 = tf.layers.dropout(h1,self.keep_prob,name='dropout')
            return tf.nn.relu(h2, 'relu')

    def _get_cost(self):
        with tf.variable_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.label, name='cross_entropy'), name='cost')

    def _get_acc(self):
        with tf.variable_scope("indicator"):
            correct_pred = tf.equal(self.predictor, tf.argmax(self.label,1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_pred, tf.float32), name='accuracy')
