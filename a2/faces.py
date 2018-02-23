from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import torch
from torch.autograd import Variable
from scipy.io import loadmat
#import cPickle
import os
from scipy.io import loadmat
import hashlib

#import pickle

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Load Images from project1
test = np.load("test.npy")
train = np.load("train.npy")
validate = np.load("validate.npy")
data = np.load("data.npy")

#====================== Part 8 ===============================
#download data and remove bad images according to hash

def download_images():

    raw_male = "facescrub_actors_male.txt"

    raw_female = "facescrub_actresses.txt"
    
    data = get_raw(raw_female, "F")
    male = get_raw(raw_male, "M")
    
    data.update(male)
    return data
    
    
act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
        
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray/255.

#testfile = urllib.request.FancyURLopener() 
  
def get_raw(textfile, gender):  
    data = {}    
    
    #create directories to save photos
    if not os.path.isdir("uncropped"):
        os.makedirs("uncropped")
    if not os.path.isdir("cropped"):
        os.makedirs("cropped")    
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(textfile):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
         
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 45)
            
                #croped out saved inmages
                dim = line.split()[5].split(",")
                x1 = int(dim[0])
                y1 = int(dim[1])
                x2 = int(dim[2])
                y2 = int(dim[3])  
                hashval = line.split()[6]
                print(filename + "----" + hashval)
                print(dim)
                #bracco0.jpg----4f9a8a6f1377b03133bc4df0cc240d924e938bc4f163703d4222c796c5d0bd92
                #['84', '44', '187', '147'] 
                #bracco1.jpg----963a9b134c26f8aff0714bbe2c11b88f7995fd89a5beb5a88bf21ae46244a850
                #['861', '971', '2107', '2217']                

                if not os.path.isfile("uncropped/"+filename):
                    continue
                elif (hashlib.sha256(open("uncropped/" + filename, "rb").read()).hexdigest()) != hashval:
                    continue #check hash values for invalid faces
                else:
                    try:
                        print(filename)
                        im = imread("uncropped/"+filename)
                        cropp = im[y1:y2, x1:x2]
                         
                        resized = imresize(cropp, (32, 32))                        
                        if len(im) > 2: #need to convert to grey
                            greyed = rgb2gray(resized)
                        else: #already greyed
                            greyed = resized
                            
                        imsave("cropped/"+filename, greyed)
                        if os.path.isfile("cropped/"+filename):
                            print(filename)
                            data[filename] = [name, gender]
                    except Exception:
                        print(filename + ":cannot read")
                i += 1
    return data
                
#seperate datasets into train, test and validate sets                         
def seperate_dataset(data):
    train = {}
    validate = {}
    test = {}
    keys = data.keys()
    splitted_data = []
    #split each actors 
    for j in range(len(act)):
        splitted_data.append({})
        name = act[j].split()[1].lower()
        for element in keys:
            if data[element][0]  == name:
                splitted_data[j][element] = data[element]
                
    gilpin = 0
    for i in range(len(splitted_data)):
        if "gilpin" in list(splitted_data[i].keys())[0]:
            gilpin = i
    
    for j in range(len(splitted_data)):
        actors_data = splitted_data[j]
        if j != gilpin:
            i = 0 #to keep track of how many pictures added for each actor
            keys_a = list(actors_data.keys())
            print(keys_a)
            while i<100 and len(keys_a)>0 :
                index = random.randint(0, len(keys_a)-1) #add pictures randomly
                element = keys_a[index]
                if i < 70:
                    print(element + "------" + str(i))
                    train.update({element:data[element]})
                    i = i + 1
                elif i < 90:
                    print(element + "------" + str(i))
                    test.update({element:data[element]})
                    i = i + 1
                elif i < 100:
                    print(element + "------" + str(i))
                    validate.update({element:data[element]})
                    i = i + 1
                keys_a.remove(element)
        elif j == gilpin:
            i = 0 #to keep track of how many pictures added for each actor
            keys_a = list(actors_data.keys())
            print(keys_a)
            while i<86 and len(keys_a)>0 :
                index = random.randint(0, len(keys_a)-1) #add pictures randomly
                element = keys_a[index]
                if i < 56:
                    print(element + "------" + str(i))
                    train.update({element:data[element]})
                    i = i + 1
                elif i < 76:
                    print(element + "------" + str(i))
                    test.update({element:data[element]})
                    i = i + 1
                elif i < 86:
                    print(element + "------" + str(i))
                    validate.update({element:data[element]})
                    i = i + 1            
    
    np.save("train.npy", train)
    np.save("validate.npy", validate)
    np.save("test.npy", test)
    return [train, validate, test]



#get data output to x and y and encode using ONE HOT encoding
def get_data(dataset):
    actork =['Bracco', 'Gilpin', 'Harmon', 'Baldwin', 'Hader', 'Carell']
    x = []
    y = []
    d = dict(dataset.flatten()[0])
    for pic in d.keys():
        if d[pic][0] == 'bracco':
            a = imread("cropped/"+pic).flatten()/255.0
            x.append(a)
            y.append([1, 0, 0, 0, 0, 0])
        elif d[pic][0] == 'gilpin':
            a = imread("cropped/"+pic).flatten()/255.0
            x.append(a)
            y.append([0, 1, 0, 0, 0, 0])
        elif d[pic][0] == 'harmon':
            a = imread("cropped/"+pic).flatten()/255.0
            x.append(a)
            y.append([0, 0, 1, 0, 0, 0])
        elif d[pic][0] == 'baldwin':
            a = imread("cropped/"+pic).flatten()/255.0
            x.append(a)
            y.append([0, 0, 0, 1, 0, 0])  
        elif d[pic][0] == 'hader':
            a = imread("cropped/"+pic).flatten()/255.0
            x.append(a)
            y.append([0, 0, 0, 0, 1, 0])  
        elif d[pic][0] == 'carell':
            a = imread("cropped/"+pic).flatten()/255.0
            x.append(a)
            y.append([0, 0, 0, 0, 0, 1])   

    return np.array(x), np.array(y)


def get_train(dataset):
    batch_xs = np.zeros((0, 32*32))
    batch_y_s = np.zeros( (0, 6))
    
    train_k =  ["train"+str(i) for i in range(6)]
    for k in range(6):
        batch_xs = np.vstack((batch_xs, ((np.array(d[train_k[k]])[:])/255.)  ))
        one_hot = np.zeros(10)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s


train_x, train_y = get_data(train)
test_x, test_y = get_data(test)

dim_x = 1024
dim_h = 20
dim_out = 6

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

x = Variable(torch.from_numpy(train_x)).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_y,1)), requires_grad=False).type(dtype_long)
#Subsample the training set for faster training
#train_idx = np.random.permutation(range(train_x.shape[0]))[:1000]
#x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
#y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)

model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)


loss_fn = torch.nn.CrossEntropyLoss()


learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step
    if t%100 == 0:
        print("iteration--" + str(t))
                       
x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_pred = model(x).data.numpy()
np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))



#split data into mini-bathces using an optimizer of ...my choice
def mini_batch(x, y):
    return x, y


#plot learning curves
def learning_curve():
    pass

#train the algorithm to classify faces
def train(size, alpha):
    #initiallize weights 
    
    
    pass

#test the performance of the training algorithm 
def test(theta):
    size = 20
    pass