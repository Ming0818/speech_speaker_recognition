# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:41:03 2015

@author: thomai
"""

import numpy as np
import matplotlib.pyplot as mat
import pickle
from tools2 import log_multivariate_normal_density_diag
from proto2 import *
tidigits = np.load('data/tidigits_examples.npz')['tidigits']
example = np.load('data/lab2_example.npz')['example'].item()
models = np.load('data/lab2_models.npz')['models']
with open('data/mfccs.dat', 'rb') as infile:
    mfccs = pickle.load(infile)


"""
    Begin main
"""
for idx in range(len(tidigits)):
    tidigits[idx]['wid'] = tidigits[idx]['gender'] + '_' + \
      tidigits[idx]['speaker'] + '_' + tidigits[idx]['digit'] + \
      tidigits[idx]['repetition'];

"""
    4 Multivariate Gaussian Density
"""
hmm_obsloglik = log_multivariate_normal_density_diag(example['mfcc'], \
  models[0]['hmm']['means'], models[0]['hmm']['covars']);

gmm_obsloglik = log_multivariate_normal_density_diag(example['mfcc'], \
  models[0]['gmm']['means'], models[0]['gmm']['covars']);


#mat.subplot(211)
#mat.imshow(np.transpose(hmm_obsloglik))
#mat.title("My hmm_obsloglik")
#
#mat.subplot(212)
#mat.imshow(np.transpose(example['hmm_obsloglik']))
#mat.title("Expected hmm_obsloglik")
#
#mat.show()
#
#mat.subplot(211)
#mat.imshow(np.transpose(gmm_obsloglik))
#mat.title("My gmm_obsloglik")
#
#mat.subplot(212)
#mat.imshow(np.transpose(example['gmm_obsloglik']))
#mat.title("Expected gmm_obsloglik")
#
#mat.show()

"""
    5 GMM Likelihood and Recognition
"""
gmm_loglik = gmmloglik(gmm_obsloglik, models[0]['gmm']['weights']);
gmm_logliks = np.zeros((len(tidigits), len(models)));

#print gmm_loglik;
#print (example['gmm_loglik']);

gmm_obslogliks = {};

""" Calculate gmm_obsloglik for all utt """
for utt in range(len(tidigits)):
    for mod in range(len(models)):
        gmm_obslogliks[utt, mod] = \
          log_multivariate_normal_density_diag(mfccs[utt], \
          models[mod]['gmm']['means'], models[mod]['gmm']['covars']);

""" Calculate gmm_loglik for all utterances from all models """
for utt in range(len(tidigits)):
    for mod in range(len(models)):
        gmm_logliks[utt, mod] = \
          gmmloglik(gmm_obslogliks[utt, mod], models[mod]['gmm']['weights']);

print gmm_logliks;

""" Classify utterances """
wrong = 0;
for utt in range(len(tidigits)):
    print "Utterance " + str(utt) + ": " + str(tidigits[utt]['digit']) + " " +\
      str(models[np.argmax(gmm_logliks[utt, :])]['digit']);
    if (tidigits[utt]['digit'] != models[np.argmax(gmm_logliks[utt, :])]['digit']):
        wrong += 1;
print "Misrecognized: " + str(wrong) + " utterances";

"""
    6 HMM Likelihood and Recognition
        6.1 Forward Algorithm
"""
logalpha = forward(hmm_obsloglik, np.log(models[0]['hmm']['startprob']), \
  np.log(models[0]['hmm']['transmat']));

hmm_loglik = logsumexp(logalpha[-1, :]);


#""" Calculate hmm_obsloglik for every utterance and every model """
#hmm_obslogliks = {};
#for utt in range(len(tidigits)):
#    for mod in range(len(models)):
#        hmm_obslogliks[utt, mod] = \
#          log_multivariate_normal_density_diag(mfccs[utt], \
#          models[mod]['hmm']['means'], models[mod]['hmm']['covars']);
#
#
#""" Use Gaussians like a GMM """
#gmm_obslogliks = {};
#for utt in range(len(tidigits)):
#    for mod in range(len(models)):
#        states = len(models[mod]['hmm']['startprob']);
#        weights = np.ones(states) / states;
#        gmm_logliks[utt, mod] = \
#          gmmloglik(hmm_obslogliks[utt, mod], weights);
#
#""" Classify utterances """
#wrong = 0;
#for utt in range(len(tidigits)):
#    print "Utterance " + str(utt) + ": " + str(tidigits[utt]['digit']) + " " +\
#      str(models[np.argmax(gmm_logliks[utt, :])]['digit']);
#    if (tidigits[utt]['digit'] != models[np.argmax(gmm_logliks[utt, :])]['digit']):
#        wrong += 1;
#print "Misrecognized: " + str(wrong) + " utterances";

#""" Calculate hmm_loglik for every utterance and every model """
#
#hmm_logliks = np.zeros((len(tidigits), len(models)));
#
#for utt in range(len(tidigits)):
#    print utt
#    for mod in range(len(models)):
#        temp = forward(hmm_obslogliks[utt, mod],\
#          np.log(models[mod]['hmm']['startprob']), \
#          np.log(models[mod]['hmm']['transmat']));
#        hmm_logliks[utt, mod] = logsumexp(temp[-1, :]);
#
#""" Classify utterances """
#wrong = 0;
#for utt in range(len(tidigits)):
#    print "Utterance " + str(utt) + ": " + str(tidigits[utt]['digit']) + " " +\
#      str(models[np.argmax(hmm_logliks[utt, :])]['digit']);
#    if (tidigits[utt]['digit'] != models[np.argmax(hmm_logliks[utt, :])]['digit']):
#        wrong += 1;
#print "Misrecognized: " + str(wrong) + " utterances";

"""
    6.2 Viterbi Approximation
"""

[best_lik, best_path] = viterbi(hmm_obsloglik, np.log(models[0]['hmm']['startprob']), \
  np.log(models[0]['hmm']['transmat']));


#mat.subplot(211)
#mat.imshow(np.transpose(logalpha))
#mat.title("My logalpha & Best Path")
#mat.plot(best_path)
#mat.grid()
##mat.colorbar();
#
#mat.subplot(212)
#mat.imshow(np.transpose(example['hmm_logalpha']))
#mat.title("Expected hmm_logalpha & Best Path")
#mat.plot(example['hmm_vloglik'][1])
#mat.grid()
##mat.colorbar()
#
#mat.show()

""" Calculate hmm_obsloglik for every utterance and every model """
hmm_obslogliks = {};
for utt in range(len(tidigits)):
    for mod in range(len(models)):
        hmm_obslogliks[utt, mod] = \
          log_multivariate_normal_density_diag(mfccs[utt], \
          models[mod]['hmm']['means'], models[mod]['hmm']['covars']);

""" Calculate best_lik for every utterance and every model """

best_liks = np.zeros((len(tidigits), len(models)));

for utt in range(len(tidigits)):
    print utt
    for mod in range(len(models)):
       [best_liks[utt, mod], temp] = viterbi(hmm_obslogliks[utt, mod], \
          np.log(models[mod]['hmm']['startprob']), \
          np.log(models[mod]['hmm']['transmat']));


""" Classify utterances """
wrong = 0;
for utt in range(len(tidigits)):
    print "Utterance " + str(utt) + ": " + str(tidigits[utt]['digit']) + " " +\
      str(models[np.argmax(best_liks[utt, :])]['digit']);
    if (tidigits[utt]['digit'] != models[np.argmax(best_liks[utt, :])]['digit']):
        wrong += 1;
print "Misrecognized: " + str(wrong) + " utterances";