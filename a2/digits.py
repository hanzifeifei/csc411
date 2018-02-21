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


#==================== Part 1 ==================================
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
    random.seed(1)
    rand = random.random(10) * len(M[train])
    r = [int(i) for i in rand]
    
    #Display 10 images for each number 
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(M[train][r[0]].reshape((28,28)), cmap=cm.gray)
    axarr[0, 1].imshow(M[train][r[1]].reshape((28,28)), cmap=cm.gray)
    axarr[0, 2].imshow(M[train][r[2]].reshape((28,28)), cmap=cm.gray)
    axarr[0, 3].imshow(M[train][r[3]].reshape((28,28)), cmap=cm.gray)   
    axarr[0, 4].imshow(M[train][r[4]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 0].imshow(M[train][r[5]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 1].imshow(M[train][r[6]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 2].imshow(M[train][r[7]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 3].imshow(M[train][r[8]].reshape((28,28)), cmap=cm.gray)
    axarr[1, 4].imshow(M[train][r[9]].reshape((28,28)), cmap=cm.gray)
        
    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)
    
    plt.show()
    
def test_part1():
    for i in range(10):
        displayNumbers(i)
    
  
#==================== Part 2 ==============================  
def calculate_output(X, W, b):
    output = dot(W.T, X) + b
    return output 

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

#==================== Part 3 ===============================

def f_p3(x, y, w, b):
    """
    Use the sum of the negative log-probabilities of all the training cases 
    as the cost function.
    """
    return -sum(y * log(softmax(calculate_output(x, w, b))))

def df_p3(x, y, w, b):
    return dot(x, (softmax(calculate_output(x, w, b)) - y).T)

#==================== Part 4 ==============================

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

def one_hot(dataset, size):
    """
    Make x and y for the one-hot encoding.
    param: size: size get from each separete dataset
    param: dataset: indicates if using "test" or "train" 
    """
    x = np.ones(size)
    y = np.array([])
    
    for i in range(10):
        data = dataset + str(i)
        x = np.vstack((x, M[data][:size]))
        y_i = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_i[i] = 1
        for j in range(size):
            y = np.vstack((y, y_i))
    
    
    return x, y
    




#if __name__ == "__main__":
    
    #PART1
    #test_part1()