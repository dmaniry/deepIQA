#!/usr/bin/python2
import numpy as np
import argparse

import chainer
import six
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import imageio
from sklearn.feature_extraction.image import extract_patches
     
from nr_model import Model
from fr_model import FRModel


parser = argparse.ArgumentParser(description='evaluate.py')
parser.add_argument('INPUT', help='path to input image')
parser.add_argument('REF', default="", nargs="?", help='path to reference image, if omitted NR IQA is assumed')
parser.add_argument('--model', '-m', default='',
                    help='path to the trained model')
parser.add_argument('--top', choices=('patchwise', 'weighted'),
                    default='weighted', help='top layer and loss definition')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')

args = parser.parse_args()

FR = True
if args.REF == "":
     FR = False

if FR:
     model = FRModel(top=args.top)
else:
     model = Model(top=args.top)

cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
serializers.load_hdf5(args.model, model)
model.to_gpu()

if FR:
     ref_img = imageio.imread(args.REF)
     patches = extract_patches(ref_img, (32,32,3), 32)
     X_ref = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))

img = imageio.imread(args.INPUT)
patches = extract_patches(img, (32,32,3), 32)
X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))


y = []
weights = []
batchsize = min(2000, X.shape[0])
t = xp.zeros((1, 1), np.float32)
for i in six.moves.range(0, X.shape[0], batchsize):
     X_batch = X[i:i + batchsize]
     X_batch = xp.array(X_batch.astype(np.float32))

     if FR:
          X_ref_batch = X_ref[i:i + batchsize]
          X_ref_batch = xp.array(X_ref_batch.astype(np.float32))
          model.forward(X_batch, X_ref_batch, t, False, n_patches_per_image=X_batch.shape[0])
     else:
          model.forward(X_batch, t, False, X_batch.shape[0])

     y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
     weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))

y = np.concatenate(y)
weights = np.concatenate(weights)

print("%f" %  (np.sum(y*weights)/np.sum(weights)))
