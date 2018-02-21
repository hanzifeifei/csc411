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

import pickle

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

snapshot = pickle.load(open("snapshot50.pkl", "rb"), encoding="latin1")

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
def calculate_output(X, W):
    output = dot(W.T, X)
    return output 

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def part_2(X, W):
    y = calculate_output(X, W)
    result = softmax(y)
    return result

#==================== Part 3 ===============================

def f_p3(x, y, w):
    """
    Use the sum of the negative log-probabilities of all the training cases 
    as the cost function.
    """
    return -sum(y * log(softmax(calculate_output(x, w))))

def df_p3(x, y, w):
    return dot(x, (softmax(calculate_output(x, w)) - y).T)

#==================== Part 4 ==============================

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    weights = []
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        weights.append(prev_t)
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print("Iter") + str(iter)
            print("Gradient: " + df(x, y, t) + "\n")
        iter += 1
    return t, weights

def one_hot(dataset, size):
    """
    Make x and y for the one-hot encoding.
    param: size: size get from each separete dataset
    param: dataset: indicates if using "test" or "train" 
    """
    x = np.empty((784, 0))
    y = np.empty((10, 0))
    
    for i in range(10):
        data = dataset + str(i)
        x = np.hstack((x, M[data][:size].T))
        y_i = np.zeros((10,1))
        y_i[i] = 1
        for j in range(size):
            y = np.hstack((y, y_i)) 
    x = x / 255.0
    x = np.vstack( (ones((1, x.shape[1])), x))
    return x, y
    
def part_4_train(alpha):
    """
    Train the neural network using gradient descent.
    """
    
    init_weights = zeros((785, 10))
    random.seed(3)
    x, y = one_hot("train", 100)

    opt_w, weights = grad_descent(f_p3, df_p3, x, y, init_weights, alpha)
    return opt_w

def test_part4(dataset, size):
    '''
    Tests performance on the training and test sets
    :param optimized_weights: thetas that will be tested
    :return: performance values in a tuple
    '''
    
    score = 0
    theta = part_4_train(0.000001)
    
    x_test, y_test = one_hot(dataset, size)
    y_pred = part_2(x_test, theta)
    
    #compare y_pred and y_test
    for i in range(size*10):
        
        if argmax(y_pred.T[i]) == argmax(y_test.T[i]):
            score += 1
    
    return score/float(size*10)

    



#if __name__ == "__main__":
    
    #PART1
    #test_part1()