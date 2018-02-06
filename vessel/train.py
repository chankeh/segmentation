#-*-coding utf-8-*-
import os
from datetime import datetime
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

def train_vessel(model,trainprovider,testprovider,model_dir,**kwargs):
    learning_rate = kwargs.get('learning_rate',1e-3)
    n_epochs = kwargs.get("n_epochs",20)
    set_size = kwargs.get('set_size',200000)
    batch_size = kwargs.get("batch_size",128)
    keep_prob = kwargs.get('keep_prob',0.5)

    test_nums = testprovider.max_index + 1   # test  image data 갯수

    train_folder_name = datetime.strftime(datetime.now(),"%y%m%d%H%M%S")
    train_folder_dir = os.path.join(model_dir,train_folder_name)
    os.makedirs(train_folder_dir,exist_ok=True) # 없으면 생성

    # logging setting
    log_path = os.path.join(train_folder_dir,"train_process.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger = logging.getLogger('vesselDNN')
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    # tensorboard summmary directory
    tensor_board_dir = os.path.join(train_folder_dir,"board")
    os.makedirs(tensor_board_dir,exist_ok=True)

    # cpkt model directory
    ckpt_dir = os.path.join(train_folder_dir,"model")
    os.makedirs(ckpt_dir,exist_ok=True)

    with tf.Session(graph=model.graph) as sess:
        ##### -----summary operator setting -----
        tf.summary.scalar("loss", model.cost)
        tf.summary.scalar("accuracy", model.accuracy)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(tensor_board_dir+'/train', sess.graph)
        test_writer = tf.summary.FileWriter(tensor_board_dir+'/test')
        learning_t = tf.placeholder(tf.float32, shape=[])
        saver = tf.train.Saver(max_to_keep=10)

        optm = tf.train.AdamOptimizer(learning_rate=learning_t).minimize(model.cost)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(init)
        meta_graph_path = os.path.join(ckpt_dir, "graph")
        saver.save(sess,meta_graph_path)
        step = 0
        for index in range(1,n_epochs+1):
            trainLoss = []; trainAcc = [];
            testLoss = []; testAcc = [];
            batch_x, batch_y = trainprovider(set_size)
            for idx in range(len(batch_x) // batch_size):
                step += 1
                if step % 5000 == 0:
                    learning_rate = learning_rate / 10
                x = batch_x[(batch_size*idx) : batch_size*(idx+1),...]
                y = batch_y[(batch_size*idx) : batch_size*(idx+1)]
                sess.run(optm, feed_dict={"input:0":x,"label:0":y,"keep_prob:0":keep_prob,learning_t:learning_rate})
                #### ----- Summarizing train step -----
                if step % 10 == 0 :
                    summary_str, loss, acc = sess.run([summary_op,model.cost,model.accuracy],
                                                 feed_dict={"input:0":x,"label:0":y,"keep_prob:0":1.})
                    train_writer.add_summary(summary_str, step)
                    trainLoss.append(loss); trainAcc.append(acc)

                    test_x, test_y = testprovider(batch_size)
                    summary_str, loss, acc = sess.run([summary_op,model.cost,model.accuracy],
                                                 feed_dict={"input:0":test_x,"label:0":test_y,"keep_prob:0":1.})
                    test_writer.add_summary(summary_str, step)
                    testLoss.append(loss); testAcc.append(acc)

            trainLoss = np.mean(trainLoss)
            trainAcc = np.mean(trainAcc)
            testLoss = np.mean(testLoss)
            testAcc = np.mean(testAcc)

            #### ---- MODEL SAVE ---- ####
            model_name = "{:2.2f}_{:03d}".format(testAcc*100,index)
            model_path = os.path.join(ckpt_dir,model_name)
            saver.save(sess,model_path,write_meta_graph=False)

            logger.info("[{:03d}/{:03d}] trainLoss: {:.4f} trainAcc: {:.4f} testLoss: {:.4f} testAcc: {:.4f}"\
                .format(index,n_epochs,trainLoss,trainAcc,testLoss,testAcc))
