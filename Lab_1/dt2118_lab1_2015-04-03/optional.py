# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:51:46 2015

@author: thomai
"""

import numpy as np
import matplotlib.pyplot as mat
import scipy.signal
import scipy.fftpack
from tools import trfbank
from sklearn.mixture import GMM
#from mfcc import mfcc
tidigits = np.load('tidigits_examples.npz')['tidigits']
example = np.load('tidigits_examples.npz')['example']

D = np.load('D.npy')
D_std = np.load('D_std.npy')
D_probs = np.load('D_probs.npy')
local_dist = np.load('locals.npy')

for idx in range(len(tidigits)):
    tidigits[idx]['wid'] = tidigits[idx]['gender'] + '_' + \
      tidigits[idx]['speaker'] + '_' + tidigits[idx]['digit'] + \
      tidigits[idx]['repetition'];



linked = scipy.cluster.hierarchy.linkage(D_probs, method='complete');
# Get labels
labels = np.empty(len(tidigits), dtype=object);
#labels = '';
for i in range(len(tidigits)):
    labels[i] = tidigits[i]['wid'];
#    labels = labels + str(i) + ' ' +  tidigits[i]['wid'] + '\n'

labels1 = np.arange(len(tidigits));



R = scipy.cluster.hierarchy.dendrogram(linked, labels=labels);
mat.xticks(rotation=90)
#mat.setp(xtickNames, rotation=45, fontsize=8)
#mat.figtext(0, 0, labels, size='small')
mat.show()