# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:40:48 2015

@author: thomai
"""
import matplotlib.pyplot as mat
import pickle

means = pickle.load(open("means.p", "rb"))
variances = pickle.load(open("variances.p", "rb"))

# Now we have all the variances and means
it_vec = range(len(means))
for obs in range(len(means[1])):
    temp = np.zeros(len(means))
    for it in range(1, len(means) + 1):
        temp[it - 1] = means[it][obs]  
    mat.plot(it_vec, temp)
mat.grid()
mat.title("Evolution of means of the second state of the phoneme 'r'")
mat.show()

it_vec = range(len(variances))
for obs in range(len(variances[1])):
    temp = np.zeros(len(variances))
    for it in range(1, len(variances) + 1):
        temp[it - 1] = variances[it][obs]  
    mat.plot(it_vec, temp)
mat.grid()
mat.title("Evolution of variances of the second state of the phoneme 'r'")
mat.show()