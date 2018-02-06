#-*-coding utf-8-*-
import os
import sys
import glob
import shutil
import yaml
import random
import logging
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("..")

import numpy as np
import tensorflow as tf


def train_model(model, trainprovider, testprovider, model_dir, **kwargs):
    learning_rate = kwargs.get('learning_rate', 0.0005)
    batch_size = kwargs.get("batch_size", 10)
    n_epochs = kwargs.get("n_epochs", 50)
    keep_prod = kwargs.get("keep_prod", 0.5)

    train_nums = trainprovider.max_index + 1  # train image data 갯수
    test_nums = testprovider.max_index + 1   # test  image data 갯수

    test_x, test_y = testprovider(test_nums)  # test data set

    train_folder_name = datetime.strftime(datetime.now(), "%y%m%d%H%M%S")
    train_folder_dir = os.path.join(model_dir, train_folder_name)
    os.makedirs(train_folder_dir, exist_ok=True)  # 없으면 생성

    # logging setting
    log_path = os.path.join(train_folder_dir, "train_process.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger = logging.getLogger('Unet')
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    # tensorboard summmary directory
    tensor_board_dir = os.path.join(train_folder_dir, "board")
    os.makedirs(tensor_board_dir, exist_ok=True)

    # cpkt model directory
    ckpt_dir = os.path.join(train_folder_dir, "model")
    os.makedirs(ckpt_dir, exist_ok=True)

    # image directory
    pred_dir = os.path.join(train_folder_dir, 'pred')
    os.makedirs(pred_dir, exist_ok=True)

    with tf.Session(graph=model.graph) as sess:
        # -----summary operator setting -----
        tf.summary.scalar("loss", model.cost)
        tf.summary.scalar("accuracy", model.accuracy)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            tensor_board_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(tensor_board_dir + '/test')

        saver = tf.train.Saver()
        optm = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model.cost)
        init = [tf.global_variables_initializer(
        ), tf.local_variables_initializer()]
        sess.run(init)
        step = 0
        for index in range(1, n_epochs + 1):
            trainLoss = []
            trainAcc = []

            for _ in range(train_nums // batch_size):
                step += 1
                x, y = trainprovider(batch_size)
                sess.run(optm, feed_dict={
                         "input:0": x, "label:0": y, "keep_prod:0": keep_prod, })
                #### ----- Summarizing train step ----- ####
                if step % 10 == 0:
                    summary_str, loss, acc = sess.run([summary_op, model.cost, model.accuracy],
                                                      feed_dict={"input:0": x, "label:0": y, "keep_prod:0": 1.})
                    train_writer.add_summary(summary_str, step)
                    trainLoss.append(loss)
                    trainAcc.append(acc)
            trainLoss = np.mean(trainLoss)
            trainAcc = np.mean(trainAcc)

            #### ---- TEST ----  ####
            testLoss = []
            testAcc = []
            case_list = []
            pad = np.ones((model.height, 10, 3))
            for idx in range(test_nums):
                x = test_x[idx:idx + 1, ...]
                y = test_y[idx:idx + 1, ...]
                summary_str, Loss, Acc, pred, arg_pred = sess.run([
                    summary_op, model.cost, model.accuracy, model.hypothesis, model.predictor],
                    feed_dict={"input:0": x, "label:0": y, "keep_prod:0": 1.})

                testLoss.append(Loss)
                testAcc.append(Acc)
                test_writer.add_summary(summary_str, step)

                ##### ----- TEST IMAGE MERGING ----- #####
                img = x[0, ...]
                ground_truth = cv2.cvtColor(y[0, ..., 1], cv2.COLOR_GRAY2RGB)
                pred_mask = cv2.cvtColor(pred[0, ..., 1].astype(
                    np.float32), cv2.COLOR_GRAY2RGB)
                arg_pred_mask = cv2.cvtColor(
                    arg_pred[0, ...].astype(np.uint8), cv2.COLOR_GRAY2RGB)
                case = np.concatenate(
                    [img, pad, ground_truth, pad, pred_mask, pad, arg_pred_mask], axis=1)
                case_list.append(case)

            testLoss = np.mean(testLoss)
            testAcc = np.mean(testAcc)

            #### ---- MODEL SAVE ---- ####
            model_name = "{:03d}_{:2.2f}".format(index, testAcc * 100)
            model_path = os.path.join(ckpt_dir, model_name)
            saver.save(sess, model_path)
            #### ---- DRAW TEST RESULT ---- ####
            cases = np.concatenate(case_list)
            pred_img_name = "{:03d}_{:2.2f}.png".format(index, testAcc * 100)
            pred_img_path = os.path.join(pred_dir, pred_img_name)
            plt.imsave(pred_img_path, cases)

            logger.info("[{:03d}/{:03d}] trainLoss: {:.4f}\
                trainAcc: {:.4f} testLoss: {:.4f} testAcc: {:.4f}"
                        .format(index, n_epochs, trainLoss, trainAcc, testLoss, testAcc))
