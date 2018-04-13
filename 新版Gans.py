import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C:/MNIST_data/')
img = mnist.train.images[50]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')


# 构建模型
def get_inputs(real_size, noise_size):
    """
    输入函数主要来定义真实图像tensor与噪声图像tensor
    """
    real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')  # 噪声图片

    return real_img, noise_img


def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    """
    生成器
    接收一个噪声图片，输出一个与真实图片一样size的图像。
    noise_img: 生成器的输入
    n_units: 隐层单元个数
    out_dim: 生成器输出tensor的size，这里应该为32*32=784
    alpha: leaky ReLU系数
    return： 生成图片对应的logits，生成的图像结果，
    """
    with tf.variable_scope("generator", reuse=reuse):
        # 全连接层
        hidden1 = tf.layers.dense(noise_img, n_units)  # 输入tensor 和输出维度

        # 采用Leaky ReLU作为激活函数的隐层
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        # dropout防止过拟合
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        # logits & outputs
        logits = tf.layers.dense(hidden1, out_dim)

        # 在输出层加入tanh激活函数
        outputs = tf.tanh(logits)

        return logits, outputs


def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    """
    判别器

    n_units: 隐层结点数量
    alpha: Leaky ReLU系数
    """

    with tf.variable_scope("discriminator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        # logits & outputs
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


# 定义参数
# 真实图像的size
img_size = mnist.train.images[0].shape[0]
# 传入给generator的噪声size
noise_size = 100
# 生成器隐层参数
g_units = 128
# 判别器隐层参数
d_units = 128
# leaky ReLU的参数
alpha = 0.01
# learning_rate
learning_rate = 0.001
# label smoothing
smooth = 0.1  # smooth是进行Label Smoothing Regularization的参数

# 构建网络

tf.reset_default_graph()
real_img, noise_img = get_inputs(img_size, noise_size)

# generator
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)

# discriminator
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units,
                                                  reuse=True)  # 真实图片与生成图片是共享参数的，因此在判别器输入生成图片时，需要reuse参数。

# Loss

"""
 discriminator的目的在于对于给定的真图片，识别为真（1），对于generator生成的图片，识别为假（0），因此它的loss包含了真实图片的loss和生成器图片的loss两部分。
 generator的目的在于让discriminator识别不出它的图片是假的，如果用1代表真，0代表假，那么generator生成的图片经过discriminator后要输出为1，因为generator想要骗过discriminator。

"""

# discriminator的loss
# 识别真实图片,在这里，我们使用了单边的Label Smoothing Regularization，它是一种防止过拟合的方式，
# 在传统的分类中，我们的目标非0即1，从直觉上来理解的话，这样的目标不够soft，会导致训练出的模型对于自己的预测结果过于自信。
# 因此我们加入一个平滑值来让判别器的泛化效果更好。

# d_loss_real对应着真实图片的loss，它尽可能让判别器的输出接近于1。
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)) * (1 - smooth))
# d_loss_fake对应着生成图片的loss，它尽可能地让判别器输出为0。
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_fake)))
# 整个判别器的loss。d_loss_real与d_loss_fake加起来就是整个判别器的损失。
d_loss = tf.add(d_loss_real, d_loss_fake)

# 整个生成器的loss。希望让判别器对自己生成的图片尽可能输出为1。
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)) * (1 - smooth))

# Optimizer
train_vars = tf.trainable_variables()

# generator中的tensor
g_vars = [var for var in train_vars if var.name.startswith("generator")]

# discriminator中的tensor
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

# Training

# batch_size
batch_size = 64
# 训练迭代轮数
epochs = 100
# 抽取样本数
n_sample = 25  # 在整个训练过程中记录了25个样本在不同阶段的samples图像，以序列化的方式进行了保存，

# 存储测试样例
samples = []
# 存储loss
losses = []
# 保存生成器变量
saver = tf.train.Saver(var_list=g_vars)

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        """对每一轮"""
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape((batch_size, 784))
            # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
            batch_images = batch_images * 2 - 1

            # generator的输入噪声
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})

        # 每一轮结束计算loss
        train_loss_d = sess.run(d_loss,
                                feed_dict={real_img: batch_images,
                                           noise_img: batch_noise})
        # real img loss，尽可能为1
        train_loss_d_real = sess.run(d_loss_real,
                                     feed_dict={real_img: batch_images,
                                                noise_img: batch_noise})
        # fake img loss，尽可能为0
        train_loss_d_fake = sess.run(d_loss_fake,
                                     feed_dict={real_img: batch_images,
                                                noise_img: batch_noise})
        # generator loss，尽可能为1
        train_loss_g = sess.run(g_loss,
                                feed_dict={noise_img: batch_noise})

        print("Epoch {}/{}...".format(e + 1, epochs),
              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real,
                                                                                  train_loss_d_fake),
              "Generator Loss: {:.4f}".format(train_loss_g))
        # 记录各类loss值
        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

        # 抽取样本后期进行观察
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                               feed_dict={noise_img: sample_noise})
        samples.append(gen_samples)

        # 存储checkpoints
        saver.save(sess, './checkpoints/%s_generator.ckpt' % e)

# 绘制LOSS曲线
fig, ax = plt.subplots(figsize=(20, 7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator Total Loss')
plt.plot(losses.T[1], label='Discriminator Real Loss')
plt.plot(losses.T[2], label='Discriminator Fake Loss')
plt.plot(losses.T[3], label='Generator')
plt.title("Training Losses")
plt.legend()

# 将sample的生成数据记录下来
with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)


def view_samples(epoch, samples):
    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]):  # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')

    return fig, axes


_ = view_samples(-1, samples)  # 显示最后一轮的outputs

# 生成新的图片

# 加载我们的生成器变量
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
    gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                           feed_dict={noise_img: sample_noise})
