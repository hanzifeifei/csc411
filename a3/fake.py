import numpy as np
import operator
import math
import random
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import tree
import graphviz 

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

#print the top words for each list
def top():
    i = -1
    while i > -10:
        print("real" + str(sorted_real[i]))
        i = i -1
    j = -1
    while j > -10:
        print("fake" + str(sorted_fake[j]))
        j = j -1    

#split the datas into train, validate, and test set by random
random.seed(0)

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
    prob_not_word_real = []
    prob_not_word_fake = []    
    for i in word_list.keys():
        #P(word_i|real)
        P_word_i_real = (word_list[i][0]+m*p)/float(count_real + 1)
        #P(word_i|fake)
        P_word_i_fake = (word_list[i][1]+m*p)/float(count_fake + 1)
        
        if i in headline:
            prob_word_real.append(P_word_i_real)
            prob_word_fake.append(P_word_i_fake)
            prob_not_word_real.append(1. - P_word_i_real)
            prob_not_word_fake.append(1. - P_word_i_fake)            
        elif i not in headline:
            prob_word_real.append(1. - P_word_i_real)
            prob_word_fake.append(1. - P_word_i_fake)
            prob_not_word_real.append(P_word_i_real)
            prob_not_word_fake.append(P_word_i_fake)            
    
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
    
    #compute P(class)*P(words|class)
    prob_real_words = prob_real * multi_real
    prob_fake_words = prob_fake * multi_fake
    
    #compute P(class)*(1 - P(words|class)) for part 3
    prob_real_not_words = prob_real * (1. - multi_real)
    prob_fake_not_words = prob_fake * (1. - multi_fake)   
    
    #probability that the given headline is fake, P(fake|words)
    prob = prob_fake_words/ (prob_fake_words + prob_real_words)
    
    #probability that the headline is fake when the word absence, P(fake|~words), for part 3
    prob_absence = prob_fake_not_words/ (prob_fake_not_words + prob_real_not_words)    
    
    result = "real"
    if prob > 0.5:
        result = "fake"
    
    return result, prob, prob_absence

def test_part2():
    m = 1
    p = 0.1
    
    count_train = 0
    n_train = len(train_real) + len(train_fake)
    for headline in train_real:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_train += 1
    for headline in train_fake:
        result, prob_fake, prob_absence= Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_train += 1
    performance_train = count_train / float(n_train) * 100
    print("The performance of the Naive Bayes classifer on the training set is " + str(performance_train) + "%")
    
    count_val = 0
    n_val = len(validate_real) + len(validate_fake)
    for headline in validate_real:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_val += 1
    for headline in validate_fake:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "fake":
            count_val += 1
    performance_val = count_val / float(n_val) * 100
    print("The performance of the Naive Bayes classifer on the validationx set is " + str(performance_val) + "%")  
    
    count_test = 0
    n_test = len(test_real) + len(test_fake)
    for headline in test_real:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
        if result == "real":
            count_test += 1
    for headline in test_fake:
        result, prob_fake, prob_absence = Naive_Bayes_classifier(headline, word_list, train_real, train_fake, m, p)
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
    
    #compute P(fake|word) and P(fake|~word) for all words
    for word in word_list.keys():
        words = np.append(words, word)
        word = list(word)
        result, prob_fake, prob_absence = Naive_Bayes_classifier(word, word_list, train_real, train_fake, 1, 0.1)
        prob_fake_word = np.append(prob_fake_word, prob_fake)
        prob_fake_not_word = np.append(prob_fake_not_word, prob_absence)
    
    #compute P(real|word) and P(real|~word) for all words
    for i in range(len(prob_fake_word)):
        prob_real_word = np.append(prob_real_word, 1. - prob_fake_word[i])
        prob_real_not_word = np.append(prob_real_not_word, 1. - prob_fake_not_word[i])
    
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
    print the last ten word in ascending array.
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






#=============== PART 7 ================================#

def part7():
    #part7a
    #-----Train the decision tree classifier, find the best parameters-----------
    training = np.append(train_real, train_fake)
    training_label = [1] * len(train_real) + [0] * len(train_fake)
    
    validation = np.append(validate_real, validate_fake)
    validation_label = [1] * len(validate_real) + [0] * len(validate_fake)
    
    test = np.append(test_real, test_fake)
    test_label = [1] * len(test_real) + [0] * len(test_fake)
    
    #assign each word a unique number and save the number of total unique words as num_words
    word_index = {}
    num_words = 0
    for headline in training:
        for word in headline:
            if word not in word_index: 
                word_index[word] = num_words
                num_words += 1
    for headline in validation:
        for word in headline:
            if word not in word_index: 
                word_index[word] = num_words
                num_words += 1
    for headline in test:
        for word in headline:
            if word not in word_index: 
                word_index[word] = num_words
                num_words += 1
    """
    make training set, validation set and testing set into 2-D numpy arrays which shows occurrence of words in each headline.
    For a single headline, the length of the array is the number of unique words. At each index of the array, it is 1 if the word is
    in the headline, it is 0 otherwise.
    """
    training_set = np.zeros((0, num_words))
    validation_set = np.zeros((0, num_words)) 
    test_set = np.zeros((0, num_words))
    
    for headline in training:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        training_set = np.vstack((training_set, i))
    for headline in validation:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        validation_set = np.vstack((validation_set, i))
    for headline in test:
        i = np.zeros(num_words)
        for word in headline:
            i[word_index[word]] = 1.
        i = np.reshape(i, [1, num_words])
        test_set = np.vstack((test_set, i))  
    
    max_depth_list = [3, 10, 20, 50, 100, 150, 200, 300, 500, 700]
    
    for depth in max_depth_list:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(training_set, training_label)
        print("Max depth is: " + str(depth))
        print("Training: " + str(100*clf.score(training_set, training_label)) + " Validation: " + str(100*clf.score(validation_set, validation_label)))
        
    #Highest accuracy is achived at max_depth 300
    clf = tree.DecisionTreeClassifier(max_depth=300)
    clf = clf.fit(training_set, training_label)    

    #part7b
    #----------visualize the first two layers of the decision tree
    word = []
    for i in word_index.keys():
        word.append(i)  
    
    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=2, filled=True, rounded=True, class_names=['fake', 'real'], feature_names=word)
    graph = graphviz.Source(dot_data)
    graph.render(view=True)
