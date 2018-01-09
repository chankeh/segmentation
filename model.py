from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf


class Unet(object):
    in_node = None
    label = None
    is_training = None
    output_map = None  # output tensor
    unet_graph = None  # graph
    cost = None
    # indicator of performance
    accuracy = None
    iou = None
    jaccard = None

    def __init__(self, **kwargs):
        # the speicification of image and label shape
        self.height = kwargs.get('height', 256)  # the height size of image
        self.width = kwargs.get('width', 256)  # the width size of image
        self.channels = kwargs.get('channels', 3)  # the channel size of image
        self.n_class = kwargs.get('n_class', 2)  # the number of output labels

        # the specification of model shape
        self.layers = kwargs.get('layers', 4)  # the depth of u-net
        # the number of first layer's feature
        self.feature_root = kwargs.get('feature_root', 64)
        # the size of convolutional filter
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.pool_size = kwargs.get('pool_size', 2)  # max-pooling size
        self.l2_scale = kwargs.get('l2_scale', 1e-4)  # l2 regularizer value

        with tf.Graph().as_default() as graph:
            # input tensor
            self.in_node = tf.placeholder(
                tf.float32, [None, self.height, self.width, self.channels], name="input")
            self.label = tf.placeholder(
                tf.float32, [None, self.height, self.width, self.n_class], name="label")
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            # l2 regularizer tensor
            if self.l2_scale is not None:
                self.regularizer = tf.contrib.layers.l2_regularizer(
                    scale=self.l2_scale)
            else:
                self.regularizer = None

            # left wing(conv) of U-Net
            dw_conv_list = []
            for layer in range(self.layers):
                features = 2**layer*self.feature_root
                with tf.variable_scope("down_{}".format(layer)):
                    res1, layer1 = self._conv_batch_relu(
                        self.in_node, None, "layer1", features=features)
                    _, layer2 = self._conv_batch_relu(
                        layer1, res1, 'layer2', features=features)
                    self.in_node = tf.layers.max_pooling2d(
                        layer2, pool_size=self.pool_size, strides=self.pool_size, padding='same', name='max_pool')
                    dw_conv_list.append(layer2)

            # bottom of U-Net
            with tf.variable_scope('middle'):
                res1, layer1 = self._conv_batch_relu(
                    self.in_node, None, 'layer1', features=features)
                _, layer2 = self._conv_batch_relu(
                    layer1, res1, 'layer2', features=features)
                self.in_node = layer2

            # right wing(deconv) of U-Net
            up_conv_list = []
            for layer in range(self.layers-1, -1, -1):
                features = 2**(layer)*self.feature_root
                with tf.variable_scope("up_{}".format(layer)):
                    deconv1 = tf.layers.conv2d_transpose(self.in_node, filters=features,
                                                         kernel_size=self.kernel_size,
                                                         strides=self.pool_size,
                                                         padding='same', activation=None,
                                                         kernel_regularizer=self.regularizer, name='deconv')
                    res1 = deconv1
                    deconv1 = tf.concat([dw_conv_list[layer], deconv1], axis=3)
                    _, layer1 = self._conv_batch_relu(
                        deconv1, None, "layer1", features=features)
                    _, layer2 = self._conv_batch_relu(
                        layer1, res1, 'layer2', features=features)
                    self.in_node = layer2

            # OUTPUT
            self.output_map = tf.layers.conv2d(self.in_node, filters=self.n_class,
                                               kernel_size=1, strides=(1, 1),
                                               padding='same', activation=None,
                                               kernel_regularizer=self.regularizer, name='conv2')
            self.unet_graph = graph
            self._get_cost()
            self._get_acc()

    def _conv_batch_relu(self, in_node, res_node, scope, features):
        with tf.variable_scope(scope):
            h1 = tf.layers.conv2d(in_node, filters=features, kernel_size=self.kernel_size,
                                  strides=(1, 1), padding='same', activation=None,
                                  kernel_regularizer=self.regularizer, name='conv')
            if res_node is not None:
                h1 = tf.add(h1, res_node, name='residual_add')
            h2 = tf.layers.batch_normalization(
                h1, axis=-1, momentum=0.99, epsilon=0.0001,
                center=True, scale=True, training=self.is_training, name='bn')
            return h1, tf.nn.relu(h2, 'relu')

    def _get_cost(self):
        with tf.variable_scope("cost"):
            flat_pred = tf.reshape(
                self.output_map, shape=[-1, self.n_class], name='flat_pred')
            flat_label = tf.reshape(
                self.label, shape=[-1, self.n_class], name='flat_label')
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=flat_pred, labels=flat_label, name='cross_entropy'), name='cost')

    def _get_acc(self):
        with tf.variable_scope("indicator"):
            correct_pred = tf.equal(
                tf.argmax(self.output_map, 3), tf.argmax(self.label, 3))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_pred, tf.float32), name='accuracy')

            flat_pred = tf.reshape(
                self.output_map, shape=[-1, self.n_class], name='flat_pred')
            flat_label = tf.reshape(
                self.label, shape=[-1, self.n_class], name='flat_label')

            # IOU : Intersection-Over-Union
            tp = tf.reduce_sum(tf.multiply(flat_label, flat_pred), 1)
            fn = tf.reduce_sum(tf.multiply(flat_label, 1-flat_pred), 1)
            fp = tf.reduce_sum(tf.multiply(1-flat_label, flat_pred), 1)
            self.iou = tf.reduce_mean((tp/(tp+fn+fp)), name='iou')
            # Jaccard Similarity
            pred_y_mul = tf.multiply(flat_pred, flat_label)
            a = tf.reduce_mean(pred_y_mul, 0)[1]
            b = tf.reduce_mean(flat_pred, 0)[1]
            c = tf.reduce_mean(flat_label, 0)[1]
            self.jaccard = tf.reduce_mean(1-(a/b+c-a), name='jaccard')
