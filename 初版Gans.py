#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2018/4/9/0009 16:32
# @Author  : Wangkang
# @File    : gans.py
# @Software: PyCharm
# @(๑╹ヮ╹๑)ﾉ Studying makes me happy

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = False
to_restore = False
output_path = "output"

# 总迭代次数500
max_epoch = 500

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256


# generate (model 1)
def build_generator(z_prior):
    """
    从随机噪声中随机挑选数据z_prior输入，输出为G(z)
    :param z_prior: z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")  256*100
    :return:
    """
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)  # 产生正太分布  100*150
    b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)



    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)  # 150*300
    b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)  #

    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)  # 300*784
    b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3  # 256*784

    x_generate = tf.nn.tanh(h3)  # 256*784
    g_params = [w1, b1, w2, b2, w3, b3]
    return x_generate, g_params


# discriminator (model 2)
def build_discriminator(x_data, x_generated, keep_prob):
    """

    :param x_data: 256*784
    :param x_generated:256*784
    :param keep_prob:
    :return:
    """
    # tf.concat
    x_in = tf.concat([x_data, x_generated], 0)
    w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)  # 784*300
    b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)  # h2_size = 300
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)  # h1 = 256*300

    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)  # 300*150
    b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)  # h1_size = 150
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)  # h2 = 256*150

    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)  # 150*1
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3  # h3 = 256*1

    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))  # 函数原型 tf.slice(inputs,begin,size,name='') 用途：从inputs中抽取部分内容,inputs：可以是list,array,tensor,begin：n维列表，begin[i] 表示从inputs中第i维抽取数据时，相对0的起始偏移量，也就是从第i维的begin[i]开始抽取数据,size：n维列表，size[i]表示要抽取的第i维元素的数目
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    d_params = [w1, b1, w2, b2, w3, b3]
    return y_data, y_generated, d_params


#
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    """

    :param batch_res: 256*784
    :param fname:
    :param grid_size:
    :param grid_pad:
    :return:
    """
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def train():
    # load data（mnist手写数据集）
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # 创建生成模型
    # 我们先从一个简单的分布中采样一个噪声信号 z（实际中可以采用[0, 1]的均匀分布或者是标准正态分布），
    # 然后经过一个生成函数后映射为我们想要的数据分布 Xg （z 和 X 都是向量）。
    x_generated, g_params = build_generator(z_prior)

    # 创建判别模型
    # 生成的数据x_generated和真实数据x_data都会输入一个识别网络 D。识别网络通过判别，输出一个标量，表示数据来自真实数据的概率。
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

    # 损失函数的设置
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))  # 判别模型的损失函数
    g_loss = - tf.log(y_generated)  # 生成模型的损失函数

    optimizer = tf.train.AdamOptimizer(1e-5)

    # 两个模型的优化函数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path) # Finds the filename of latest saved checkpoint file.
        saver.restore(sess, chkpt_fname)  # Restores previously saved variables.
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)  # 256*100

    steps = 60000 / batch_size
    for i in range(sess.run(global_step), max_epoch):  # i in range(0,500)
        """按照500次迭代，每次迭代产生一张手写体图片，然后进行判别反馈，这样持续下去，可以看到不同迭代次数的效果。"""
        for j in np.arange(steps):

            #for j in range(steps):
            print("epoch:%s, iter:%s" % (i, j))

            # 每一步迭代，我们都会加载256个训练样本，然后执行一次train_step
            x_value, _ = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)  # 256*100

            sess.run(d_trainer, feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})

            if j % 1 == 0:
                sess.run(g_trainer, feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})

        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, "output/sample{0}.jpg".format(i))

        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))

        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)   # z_test_value = 256*100
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, "output/test_result.jpg")


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()
