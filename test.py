
import numpy as np
import theano
import lasagne
import model as M
import tar

def load_trained():
    params = np.load('trained_model.npz')
    for p in lasagne.layers.get_all_params(M.model):
        p.set_value(params[p.name])

if __name__=='__main__':
    load_trained()
    x = theano.tensor.TensorType('float32', [False]*5)('x')
    deterministic_output = theano.function([x], lasagne.layers.get_output(M.model, x, deterministic=True))
    predictions, actual = [], []

    for x, y in tar.get_data(M.ROTATIONS):
        prediction = np.argmax(np.sum(deterministic_output(x), 0))
        predictions += [prediction]
        actual += [y[0]]

    print('Test accuracy = ' + str(np.mean(np.asarray(prediction, dtype=np.int) == np.asarray(actual, dtype=np.int)).mean()))
