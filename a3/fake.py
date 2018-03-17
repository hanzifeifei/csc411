import numpy as np
import operator
import math
import random
from build_sets import *

#=============== PART 1 ================================#

#get all the lines
real = []
words_real = []
for line in open("clean_real.txt"): #get real titles\
    real.append(np.array(line.split()))
real = np.array(real)


fake = []
words_fake = []
for line in open("clean_fake.txt"): #get fake titles\
    fake.append(line.split()) 
fake = np.array(fake)


words_real = []
for headline in real:
    words_real = words_real + list(set(headline))
words_real = np.array(words_real)

words_fake = []
for headline in fake:
    words_fake = words_fake + list(set(headline))
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
def top():
    i = -1
    while i > -16:
        print("real" + str(sorted_real[i]))
        i = i -1
    j = -1
    while j > -16:
        print("fake" + str(sorted_fake[j]))
        j = j -1    

#split the datas into train, validate, and test set by random
random.seed(42)

random.shuffle(fake)
random.shuffle(real)

train_real = real[:int(math.floor(len(real) * 0.7))]
train_fake = fake[:int(math.floor(len(fake) * 0.7))]

validate_real = real[int(math.ceil(len(real) * 0.7)):int(math.floor(len(real) * 0.85))]
validate_fake = fake[int(math.ceil(len(fake) * 0.7)):int(math.floor(len(fake) * 0.85))]

test_real = real[int(math.ceil(len(real) * 0.85)):]
test_fake = fake[int(math.ceil(len(fake) * 0.85)):]


#=============== PART 2 ================================#

#making a word list with the word as key and 
#number of occurences in real and fake headlines as the value
word_list = {}
for i in range(len(train_real)):
    headline = set(train_real[i])
    for word in headline:
        if word not in word_list:
            word_list[word] = [1, 0]
        else:
            word_list[word] = [word_list[word][0] + 1, word_list[word][1]]
for i in range(len(train_fake)):
    headline = set(train_fake[i])
    for word in headline:
        if word not in word_list:
            word_list[word] = [0, 1]
        else:
            word_list[word] = [word_list[word][0], word_list[word][1] + 1]


#def Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p):
def Naive_Bayes_classifier(headline, word_list, training_set, training_label, m, p):
    """
    predicting whether a headline is real or fake.
    """
    #headline = headline.split()
    
    #calculate P(fake) and P(real)
    n = len(train_real) + len(train_fake)
    count_real = len(train_real)
    count_fake = len(train_fake)
    prob_fake = len(train_fake) / float(n)
    prob_real = 1.0 - prob_fake
    
    prob_word_real = []
    prob_word_fake = []
    for i in word_list.keys():
        P_word_i_real = (word_list[i][0]+m*p)/float(count_real + 1)
        P_word_i_fake = (word_list[i][1]+m*p)/float(count_fake + 1)
        
        if i in headline:
            prob_word_real.append(P_word_i_real)
            prob_word_fake.append(P_word_i_fake)
        elif i not in headline:
            prob_word_real.append(1. - P_word_i_real)
            prob_word_fake.append(1. - P_word_i_fake)
    
    #conditional independence is assumed by Naive Bayes
    #do multiplication
    multi_real = 0
    for p in prob_word_real:
        multi_real += math.log(p)
    multi_real = math.exp(multi_real)
    
    multi_fake = 0
    for p in prob_word_fake:
        multi_fake += math.log(p)
    multi_fake = math.exp(multi_fake)
    
    prob_real_words = prob_real * multi_real
    prob_fake_words = prob_fake * multi_fake
    
    result = "real"
    #probability that the given headline is fake
    if (prob_fake_words + prob_real_words) == 0:
        result = "fake"
        return result, 0.0
    else:
        prob = prob_fake_words/ (prob_fake_words + prob_real_words)
        
        
        if prob > 0.5:
            result = "fake"
        
        return result, prob

def test_part2():
    m = 1
    p = 0.1
    
    count_train = 0
    n_train = len(train_real) + len(train_fake)
    for headline in train_real:
        result, prob_fake = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_train += 1
    for headline in train_fake:
        result, prob_fake = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_train += 1
    performance_train = count_train / float(n_train) * 100
    print("The performance of the Naive Bayes classifer on the training set is " + str(performance_train) + "%")
    
    count_val = 0
    n_val = len(validate_real) + len(validate_fake)
    for headline in validate_real:
        result, prob_fake = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_val += 1
    for headline in validate_fake:
        result, prob_fake = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_val += 1
    performance_val = count_val / float(n_val) * 100
    print("The performance of the Naive Bayes classifer on the validationx set is " + str(performance_val) + "%")  
    
    count_test = 0
    n_test = len(test_real) + len(test_fake)
    for headline in test_real:
        result, prob_fake = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_val += 1
    for headline in test_fake:
        result, prob_fake = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_val += 1
    performance_test = count_val / float(n_val) * 100
    print("The performance of the Naive Bayes classifer on the test set is " + str(performance_test) + "%")     
    
