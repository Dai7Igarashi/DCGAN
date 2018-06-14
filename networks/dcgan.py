# coding: UTF-8
'''
https://github.com/rystylee/chainer-dcgan-mnist/blob/master/net_mnist.py
を参考
'''

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L

# 学習しやすいようにノイズ加える
def add_noise(h, sigma=0.2):
    # numpy or cupy 自動判別
    xp = cuda.get_array_module(h.data)
    # 学習フェーズか自動で判断
    if chainer.config.train:
        # 標準正規分布による h.shapeの行列
        return h + sigma * xp.ramdom.randn(*h.shape)
    else:
        return h

class Generator(chainer.Chain):
    def __init__(self, n_hidden, bottom_width=3, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        # 入力ノイズの次元
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch, initialW=w)
            # deconv2d(in_channels, out_channels, ksize, stride, pad)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 2, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 2, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 2, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 1, 3, 3, 1, initialW=w)
            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        # [-1, 1)の一様乱数生成
        # 入力ノイズデータ -> shape(batchsize, data_size, 1, 1)
        # 普通は shape(batchsize, channel, dim_x, dim_y) ???
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(np.float32)

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.l0(z))), (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.sigmoid(self.dc4(h))

        return x

class Discriminator(chainer.Chain):
    def __init__(self, bottom_width=3, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()

        with self.init_scope():
            # conv2d(in_channel(grayscale=1), out_channel, ksize, stride, pad)
            self.c0 = L.Convolution2D(1, ch // 8, 3, 3, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 2, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 2, 2, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch, 2, 2, 1, initialW=w)
            self.l4 = L.Linear(None, 1, initialW=w)
            self.bn1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3 = L.BatchNormalization(ch, use_gamma=False)

    def __call__(self, x):
        # h = add_noise(x)
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        y = self.l4(h)

        return y