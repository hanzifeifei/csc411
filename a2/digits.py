from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle
import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

def part1():
    # M - is a dictionary composed of 10 training set and 10 testing sets
    # each set contains a number from 0 to 9
    
    print(len(M["train5"]))
    print(len(M["train4"]))
    #each training set have around 5000 to 7000 number images 
    #these images is an array of size 784 (28 * 28)
    
    print(len(M["test7"]))
    print(len(M["test0"]))
    #each testing set have around 900 to 1200 number images 
    #these images is an array of size 784 (28 * 28)

    for i in range(10):
        display(i)


def displayNumbers(i):
    
    train = "train" + str(i)
    
    #Display 10 images for each number 
    f, axarr = plt.subplots(5, 2)
    axarr[0, 0].imshow(M[train][150].reshape((28,28)), cmap=cm.gray)
    axarr[0, 1].imshow(M[train][160].reshape((28,28)), cmap=cm.gray)
    axarr[1, 0].imshow(M[train][170].reshape((28,28)), cmap=cm.gray)
    axarr[1, 1].imshow(M[train][180].reshape((28,28)), cmap=cm.gray)   
    axarr[2, 0].imshow(M[train][190].reshape((28,28)), cmap=cm.gray)
    axarr[3, 0].imshow(M[train][110].reshape((28,28)), cmap=cm.gray)
    axarr[4, 0].imshow(M[train][120].reshape((28,28)), cmap=cm.gray)
    axarr[2, 1].imshow(M[train][130].reshape((28,28)), cmap=cm.gray)
    axarr[3, 1].imshow(M[train][140].reshape((28,28)), cmap=cm.gray)
    axarr[4, 1].imshow(M[train][100].reshape((28,28)), cmap=cm.gray)
        
    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)
    
    plt.show()    
    
