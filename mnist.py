#!/opt/local/bin/python

import os,sys,struct,time,random
from array import array
import numpy as np

MNIST_DAT_DIR = 'XXX'

def load_labels(fn):
    f = open(os.path.join(MNIST_DAT_DIR,fn),'rb')
    xxx,n = struct.unpack('>ii',f.read(8))
    x = f.read()
    f.close()
    return array('B',x)

def load_images(fn):
    f = open(os.path.join(MNIST_DAT_DIR,fn),'rb')
    xxx,n = struct.unpack('>ii',f.read(8))
    nr,nc = struct.unpack('>ii',f.read(8))
    assert(nr==nc==28)
    nx = nr*nc
    x = f.read()
    f.close()
    return [array('B',x[i*nx:(i+1)*nx]) for i in range(n)]

LABELS = load_labels('train-labels.idx1-ubyte')
IMAGES = load_images('train-images.idx3-ubyte')
assert(len(LABELS)==len(IMAGES))

LABELS_TEST = load_labels('t10k-labels.idx1-ubyte')
IMAGES_TEST = load_images('t10k-images.idx3-ubyte')
assert(len(LABELS_TEST)==len(IMAGES_TEST))

def random_image():
    return image(random.randint(0,59999))

def label(i): return LABELS[i]
def image(i): return IMAGES[i]
def pic(i): print label(i); pic_x(image(i))

def label_test(i): return LABELS_TEST[i]
def image_test(i): return IMAGES_TEST[i]
def pic_test(i): print label_test(i); pic_x(image_test(i))

def pic_x(x):
    n = 28
    def _p(val):
        if val < 100: return ' '
        return '#'
    print '\n'.join(''.join(_p(x[i*n+j]) for j in range(n)) for i in range(n))

if __name__=='__main__':
    for ii in range(10):
        i = random.randint(0,59999)
        pic(i)
        time.sleep(1)



