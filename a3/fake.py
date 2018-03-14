import numpy as np

real = []
words_real = []
for line in open("clean_real.txt"): #get real titles\
    real.append(np.array(line.split()))
    words_real = words_real +line.split()

real = np.array(real)
words_real = np.array(words_real)
unique_real, counts_real = np.unique(words_real, return_counts=True)
real_freq = dict(zip(unique_real, counts_real))

fake = []
for line in open("clean_fake.txt"): #get fake titles\
    fake.append(line.split())

fake = np.array(fake)