import numpy as np
import theano
import theano.tensor as T
import lasagne
import model as M
import tar

def save_trained():
    layer_params = lasagne.layers.get_all_params(M.model)
    np.savez_compressed('trained_model.npz', **{p.name : p.get_value(borrow=False) for p in layer_params})
    
if __name__=='__main__':
    batch_idx = T.iscalar('batch_index')
    out_shape = lasagne.layers.get_output_shape(M.model)
    batch_slc = slice(batch_idx*M.BATCH_SIZE, (batch_idx+1)*M.BATCH_SIZE)

    layer_params = lasagne.layers.get_all_params(M.model, unwrap_shared=False)
    learn_rate = theano.shared(np.float32(M.LEARNING_RATE[0]))

    x = T.TensorType('float32', [False]*5)('x')
    y = T.TensorType('int32', [False]*1)('y')
    deterministic_output = lasagne.layers.get_output(M.model, x, deterministic=True)
    prediction = T.argmax(deterministic_output, axis=1 )
    regularization_loss = (T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(
            lasagne.layers.get_output(M.model, x)), y)), 'float32')
        + M.REGULARIZATION * lasagne.regularization.regularize_network_params(M.model, lasagne.regularization.l2))

    xx = lasagne.utils.shared_empty(5, dtype='float32')
    yy = lasagne.utils.shared_empty(1, dtype='float32')

    update = theano.function([batch_idx], regularization_loss,
        updates=lasagne.updates.momentum(regularization_loss, layer_params, learn_rate, M.MOMENTUM), 
        givens={x: xx[batch_slc], y: T.cast(yy[batch_slc], 'int32')})

    accuracy = theano.function([batch_idx], T.cast(T.mean(T.neq(prediction, y)), 'float32'),
        givens={x: xx[batch_slc], y: T.cast(yy[batch_slc], 'int32')})

    i = 0
    for _ in xrange(M.EPOCHS):
        for x, y in tar.get_data(M.BATCH_SIZE*M.BATCHES):
            num_batches = len(x)//M.BATCH_SIZE
            xx.set_value(x, borrow=True)
            yy.set_value(y, borrow=True)

            for bi in xrange(num_batches):
                acc = update(bi) # loss
                err = accuracy(bi) # accuracy
                print("iteration: " + str(i) + " error: " + str(err) + " accuracy: " + str(acc))
                i += 1

            if i > 0:
                learning_to = M.LEARNING_RATE[M.RATE_ORDER[np.searchsorted(M.RATE_ORDER, i)-1]]
                if not np.allclose(np.float32(learn_rate.get_value()), learning_to):
                    learn_rate.set_value(np.float32(learning_to))
    save_trained()
