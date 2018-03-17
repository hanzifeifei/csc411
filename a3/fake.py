import numpy as np
import operator
import math
import random
from build_sets import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
        #P(word_i|real)
        P_word_i_real = (word_list[i][0]+m*p)/float(count_real + 1)
        #P(word_i|fake)
        P_word_i_fake = (word_list[i][1]+m*p)/float(count_fake + 1)
        
        if i in headline:
            prob_word_real.append(P_word_i_real)
            prob_word_fake.append(P_word_i_fake)
        elif i not in headline:
            prob_word_real.append(1. - P_word_i_real)
            prob_word_fake.append(1. - P_word_i_fake)
    
    #conditional independence is assumed by Naive Bayes
    #do multiplication to get P(words|real) and P(words|fake)
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
    
    
    #probability that the given headline is fake, P(fake|words)
    prob = prob_fake_words/ (prob_fake_words + prob_real_words)
    
    result = "real"
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
            count_test += 1
    for headline in test_fake:
        result, prob_fake = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_test += 1
    performance_test = count_test / float(n_test) * 100
    print("The performance of the Naive Bayes classifer on the test set is " + str(performance_test) + "%")     
    
#=============== PART 3 ================================#
#part 3a

def part3a(word_list):
    """
    compute P(real|word), P(real|~word), P(fake|word), P(fake|~word), and print the top ten for each
    """
    
    words = np.array([])
    prob_real_word = np.array([])
    prob_real_not_word = np.array([])
    prob_fake_word = np.array([])
    prob_fake_not_word = np.array([])
    
    #compute P(fake|word) for all words
    for word in word_list.keys():
        words = np.append(words, word)
        word = list(word)
        result, prob_fake = Naive_Bayes_classifier(word, word_list, train_real, train_fake, 1, 0.1)
        prob_fake_word = np.append(prob_fake_word, prob_fake)
    
    #compute P(real|word) for all words
    for i in range(len(prob_fake_word)):
        prob_real_word = np.append(prob_real_word, 1 - prob_fake_word[i])
        
    #compute P(fake|~word) for all words
    for word in words:
        #P(fake|~word) = P(fake|all words) - P(fake|word)
        P_fake_words = np.sum(prob_fake_word)
        P_fake_word = prob_fake_word[np.where(words == word)]
        prob_fake_not_word = np.append(prob_fake_not_word, (P_fake_words - P_fake_word)/P_fake_words)
    
    #compute P(real|~word) for all words
    for word in words:
        #P(real|~word) = P(real|all words) - P(real|word)
        P_real_words = np.sum(prob_real_word)
        P_real_word = prob_real_word[np.where(words == word)]
        prob_real_not_word = np.append(prob_real_not_word, (P_real_words - P_real_word)/P_real_words)
    
    # print(words.shape, P_fake_given_word.shape, P_fake_given_not_word.shape, P_real_given_word.shape, P_real_given_not_word.shape)
    
    real_presence = dict(zip(words, prob_real_word))
    real_absence = dict(zip(words, prob_real_not_word))
    fake_presence = dict(zip(words, prob_fake_word))
    fake_absence = dict(zip(words, prob_fake_not_word))
    
    real_presence = sorted(real_presence.items(), key=operator.itemgetter(1))
    real_absence = sorted(real_absence.items(), key=operator.itemgetter(1))
    fake_presence = sorted(fake_presence.items(), key=operator.itemgetter(1))
    fake_absence = sorted(fake_absence.items(), key=operator.itemgetter(1))
       
    print("List the 10 words whose presence most strongly predicts that the news is real:")
    top10(real_presence)
    print("List the 10 words whose absence most strongly predicts that the news is real:")
    top10(real_absence)
    print("List the 10 words whose presence most strongly predicts that the news is fake:")
    top10(fake_presence)
    print("List the 10 words whose absence most strongly predicts that the news is fake:")
    top10(fake_absence)

def top10(rank):
    """
    print the top ten word in an array
    """
    i = -1
    while i > -11:    
        print(rank[i])
        i -= 1
        
#part3b
def part3b():
    for word in ENGLISH_STOP_WORDS:
        if word in word_list: 
            del word_list[word]
    
    part3a(word_list)
    
    
#=============== PART 4 ================================#