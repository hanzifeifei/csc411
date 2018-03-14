import numpy as np
import operator
import math

#===============PART 1================================#

#get all the lines
real = []
words_real = []
for line in open("clean_real.txt"): #get real titles\
    real.append(np.array(line.split()))
    words_real = words_real +line.split()

real = np.array(real)
words_real = np.array(words_real)

fake = []
words_fake = []
for line in open("clean_fake.txt"): #get fake titles\
    fake.append(line.split())
    words_fake = words_fake +line.split()

    
fake = np.array(fake)
words_fake = np.array(words_fake)

#get the count of each word in the lines read
unique_real, counts_real = np.unique(words_real, return_counts=True)
real_freq = dict(zip(unique_real, counts_real))
unique_fake, counts_fake = np.unique(words_fake, return_counts=True)
fake_freq = dict(zip(unique_fake, counts_fake))
    
#sort the dictionary by frequency
sorted_real = sorted(real_freq.items(), key=operator.itemgetter(1))
sorted_fake = sorted(fake_freq.items(), key=operator.itemgetter(1))

#print the top three words for each list
def topTHREE():
    i = -1
    while i > -4:
        print(sorted_real[i])
        print(sorted_fake[i])
        i = i -1

#split the datas into train, validate, and test set by random??
train_real = real[:int(math.floor(len(real) * 0.7))]
train_fake = fake[:int(math.floor(len(fake) * 0.7))]

test_real = real[int(math.ceil(len(real) * 0.7)):int(math.floor(len(real) * 0.85))]
test_fake = fake[int(math.ceil(len(fake) * 0.7)):int(math.floor(len(fake) * 0.85))]

validate_real = real[int(math.ceil(len(real) * 0.85)):]
validate_fake = fake[int(math.ceil(len(fake) * 0.85)):]