import numpy as np

import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import serializers

cuda.check_cuda_available()
xp = cuda.cupy

class Model(chainer.Chain):


    def __init__(self, top = "patchwise"):
        super(Model, self).__init__(
            conv1 = L.Convolution2D(3, 32, 3, pad=1),
            conv2 = L.Convolution2D(32, 32, 3, pad=1),
            
            conv3 = L.Convolution2D(32, 64, 3, pad=1),
            conv4 = L.Convolution2D(64, 64, 3, pad=1),

            conv5 = L.Convolution2D(64, 128, 3, pad=1),
            conv6 = L.Convolution2D(128, 128, 3, pad=1),
            
            conv7 = L.Convolution2D(128, 256, 3, pad=1),
            conv8 = L.Convolution2D(256, 256, 3, pad=1),
            
            conv9 = L.Convolution2D(256, 512, 3, pad=1),
            conv10 = L.Convolution2D(512, 512, 3, pad=1),

            fc1     = L.Linear(512, 512),
            fc2     = L.Linear(512, 1),
            
            fc1_a   = L.Linear(512, 512),
            fc2_a   = L.Linear(512, 1)
        )

        self.top = top

    def forward(self, x_data, y_data, train=True, n_patches=32):

        if not isinstance(x_data, Variable):
            x = Variable(x_data)
        else:
            x = x_data
            x_data = x.data
        self.n_images = y_data.shape[0]
        self.n_patches = x_data.shape[0]
        self.n_patches_per_image = self.n_patches / self.n_images

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h,2)
        
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h,2)
        
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(h,2)
        
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pooling_2d(h,2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pooling_2d(h,2)
        
        h_ = h
        self.h = h_

        h = F.dropout(F.relu(self.fc1(h_)), ratio=0.5)
        h = self.fc2(h)
        
        if self.top == "weighted":
            a = F.dropout(F.relu(self.fc1_a(h_)), ratio=0.5)
            a = F.relu(self.fc2_a(a))+0.000001
            t = Variable(y_data)
            self.weighted_loss(h, a, t)
        elif self.top == "patchwise":
            a = Variable(xp.ones_like(h.data))
            t = Variable(xp.repeat(y_data, n_patches))
            self.patchwise_loss(h, a, t)


        if train:
            return self.loss
        else:
            return self.loss, self.y


    def patchwise_loss(self, h, a, t):
        self.loss = F.sum(abs(h - F.reshape(t, (-1,1)))) 
        self.loss /= self.n_patches
        if self.n_images > 1:
            h = F.split_axis(h, self.n_images, 0)
            a = F.split_axis(a, self.n_images, 0)
        else:
            h, a = [h], [a]
        self.y = h
        self.a = a

    def weighted_loss(self, h, a, t):
        self.loss = 0
        if self.n_images > 1:
            h = F.split_axis(h, self.n_images, 0)
            a = F.split_axis(a, self.n_images, 0)
            t = F.split_axis(t, self.n_images, 0)
        else:
            h, a, t = [h], [a], [t]

        for i in range(self.n_images):
            y = F.sum(h[i]*a[i], 0) / F.sum(a[i], 0)
            self.loss += abs(y - F.reshape(t[i], (1,)))
        self.loss /= self.n_images
        self.y = h
        self.a = a
