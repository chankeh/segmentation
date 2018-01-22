import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

class VesselDNN(object):
    in_node = None
    label = None
    is_training = None
    keep_prod = None

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
            self.keep_prod = tf.placeholder(tf.float32, name='keep_prod')

            xavier_init = tf.contrib.layers.xavier_initializer_conv2d()

            # l2 regularizer tensor
            if self.l2_scale is not None:
                self.regularizer = tf.contrib.layers.l2_regularizer(
                    scale=self.l2_scale)
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
            fc1 = self._fc_dropout_relu(conv4, 'fc1', 512)
            fc2 = self._fc_dropout_relu(fc1, 'fc2', 512)

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
            h2 = tf.layers.dropout(h1,self.keep_prod,name='dropout')
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
