import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename



def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()




if __name__ == '__main__':
    # sess = tf.Session()
    # mnist  = input_data.read_data_sets('MNIST_data/', one_hot=True)
    # print(mnist.train.images.shape)
    # a = tf.constant(10)
    #
    # b= tf.constant(12)
    #
    # sess.run(a+b)
    # print(b)
    filename = maybe_download('text8.zip', 31344016)
    text = read_data(filename)
    print('Data size %d' % len(text))