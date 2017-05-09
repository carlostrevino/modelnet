import numpy as np
import model as M
import cStringIO as StringIO
import tarfile
import zlib

def jitter(src):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    rand = lambda: np.random.random_integers(-(M.JITTER), M.JITTER)
    for i, shift in enumerate([rand(), rand(), rand()]):
        if shift != 0: dst = np.roll(dst, shift, i+2)
    return dst

def make_array(x, y):
    return (2.0*jitter(x) - 1.0, np.asarray(y, dtype=np.float32))

def pad_array(x, y, size):
    x = x[:size]
    x[len(y):] = x[:(size-len(y))]
    y = y + y[:(size-len(y))]

def get_data(chunk):
    x, y = np.zeros((chunk, M.CHANNELS,)+M.DIMENSIONS, dtype=np.float32), []
    tar = tarfile.open('shapenet_train.tar', 'r|')

    for i, entry in enumerate(tar):
        fileobj = tar.extractfile(entry)
        name = entry.name[5:-6]
        x[i % chunk] = np.load(StringIO.StringIO(zlib.decompress(fileobj.read()))).astype(np.float32)
        y += [int(name.split('.')[0])-1]

        if chunk == len(y):
            yield make_array(x, y)
            y = []
            x.fill(0)

    if len(y) > 0:
        if len(y)%M.BATCH_SIZE:
            pad_array(x, y, int(np.ceil(len(y)/float(M.BATCH_SIZE)))*M.BATCH_SIZE)

        yield make_array(x, y)