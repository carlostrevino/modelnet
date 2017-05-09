
import numpy as np

import lasagne
import lasagne.layers
import lasagne.utils
import theano.tensor as T

BATCH_SIZE = 32
RATE_ORDER = [0, 60000, 400000, 600000]
LEARNING_RATE = {0:.001, 60000:.0001, 400000:.00005, 600000:.00001}
REGULARIZATION = .001
DIMENSIONS = (32, 32, 32)
CHANNELS = 1
CLASSES = 40
BATCHES = 64
EPOCHS = 1 # 70
JITTER = 2
ROTATIONS = 12

class RELU(lasagne.init.Initializer):
    def sample(self, shape):
        std = np.sqrt(2. / (shape[1] * np.prod(shape[2:])))
        return lasagne.utils.floatX(np.random.normal(0, std, size=shape))

def init():
    inpt = lasagne.layers.InputLayer((None, CHANNELS) + DIMENSIONS)
    conv = lasagne.layers.Conv3DLayer(inpt, 32, (5,5,5), (2,2,2), W=RELU(), nonlinearity=lambda x: T.maximum(x * .1, x), name='conv1')
    drop = lasagne.layers.DropoutLayer(conv, .2)
    conv = lasagne.layers.Conv3DLayer(drop, 32, (3,3,3), W=RELU(), nonlinearity=lambda x: T.maximum(x * .1, x), name='conv2')
    pool = lasagne.layers.MaxPool3DLayer(conv, (2,2,2))
    drop = lasagne.layers.DropoutLayer(pool, .3)
    fc   = lasagne.layers.DenseLayer(drop, 128, lasagne.init.Normal(std=.01), name='fc1')
    drop = lasagne.layers.DropoutLayer(fc, .4)
    fc   = lasagne.layers.DenseLayer(drop, CLASSES, lasagne.init.Normal(std=.01), nonlinearity=None, name='fc2')

    return fc

model = init()
