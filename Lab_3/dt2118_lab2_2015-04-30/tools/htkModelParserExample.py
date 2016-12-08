#!/usr/bin/python
# Example on how to use Georgi Dzhambazov's HTK model parser to analyse the parameters of
# HMM models. The script just outputs some information about the input model, but should
# be used mainly to learn how to handle the Hmm class and its content.
# Used in Lab 3 in DT2118 Speech and Speaker Recognition
#
# Usage:
# ./htkModelParserExample.py hmmdefinition
# Example:
# tools/htkModelParserExample.py models_MFCC/hmm7/hmmdefs.mmf
#
# (C) 2015 Giampiero Salvi <giampi@kth.se>
import sys
import pickle
import matplotlib.pyplot as mat
from matplotlib.legend_handler import HandlerLine2D
sys.path.append("htkModelParser/")
from ply import yacc
import scipy.fftpack
import htk_lexer
import htk_parser
from htk_models import *
import numpy as np


hmmfile = sys.argv[1]    
#print('Example usage for Georgi Dzhambazov\'s HTK model parser')
#print('opening HTK model file: '+hmmfile)
file = open(hmmfile)
#print('parsing the data')
data = file.read()
hmms = yacc.parse(data)

#print('the number of HMM models is: ' + str(len(hmms)))

num_hmms = len(hmms)
#mat.subplot(212)
for m in range(num_hmms):
    
    thishmm = hmms[m]
#    print('model number '+str(m+1)+' is called '+thishmm.name+' and has '+str(len(thishmm.states))+' states')
    if thishmm.name == 's' or thishmm.name == 'ao' or thishmm.name == 'sil':    
        s=1
        thisstate = thishmm.states[s][1]
    #    print('state number '+str(s+1)+' has '+str(len(thisstate.mixtures))+' mixture component(s)')
        thiscomponent = thisstate.mixtures[0][2]
    #    print('the zero\'th Gaussian term has:')
        thismean = thiscomponent.mean.vector
        x = range(len(thismean))
        thismean.insert(0, thismean.pop())
    #    print('...mean of length '+str(len(thismean))+':')
        # Cosine transform
        thismean = np.array(thismean)
        thismean = thismean.astype(float)
        mfcc = np.zeros((1, 13))
        mfcc = scipy.fftpack.realtransforms.idct(thismean, type=2, norm='ortho')
        line1, = mat.plot(x, mfcc, label=thishmm.name)
    
# Place a legend to the right of this smaller figure.
mat.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
mat.grid()
mat.show()
#    print(thismean)
#thisvar = thiscomponent.var.vector

#print('...and variance of length '+str(len(thisvar))+':')
#print(thisvar)