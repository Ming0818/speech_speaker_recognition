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
sys.path.append("htkModelParser/")
from ply import yacc
import htk_lexer
import htk_parser
from htk_models import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from matplotlib.legend_handler import HandlerLine2D

hmmfile = sys.argv[1] + '/hmmdefs.mmf'
i = 0
k = 0
n = 3
means= np.zeros((n,13))
varis= np.zeros((n,13))
idctmeans = np.zeros((n,13))
names = ['','','']

#print('Example usage for Georgi Dzhambazov\'s HTK model parser')
print('opening HTK model file: '+ hmmfile)
file = open(hmmfile)
print('parsing the data')
data = file.read()
hmms = yacc.parse(data)
#print('the number of HMM models is: ' + str(len(hmms)))
n_models = len(hmms)

while i < n_models:
    thishmm = hmms[i]
    #print('model number '+str(m+1)+' is called '+thishmm.name+' and has '+str(len(thishmm.states))+' states')
    if thishmm.name == 's' or thishmm.name == 'ao' or thishmm.name == 'sil':
        s=1
        thisstate = thishmm.states[s][1]
        #print('state number '+str(s+1)+' has '+str(len(thisstate.mixtures))+' mixture component(s)')
        thiscomponent = thisstate.mixtures[0][2]
        #print('the zero\'th Gaussian term has:')
        thismean = thiscomponent.mean.vector
        #print('...mean of length '+str(len(thismean))+':')
        #print(thismean)
        thisvar = thiscomponent.var.vector
        #print('...and variance of length '+str(len(thisvar))+':')
        #print(thisvar)
        thismean.insert(0, thismean.pop())
        thisvar.insert(0, thisvar.pop())
        means[k] = thismean
        varis[k] = thisvar
        names[k] = thishmm.name
        idctmeans[k] = scipy.fftpack.realtransforms.idct(thismean, type=2, norm='ortho')
        k = k + 1
    i = i + 1

##Plot

ax = np.linspace(1,13,13)

i = 0
color = ['.r','.b','.g','.y','.k','or','ob','og','oy','ok','*r','*b','*g']
plt.figure(1)


while i < n:
	line1, = plt.plot(ax, means[i], color[i] + '-', label=names[i])
	plt.hold(True)
	i = i + 1


plt.title('MEANS')
plt.ylabel('Phoneme')
plt.xlabel('Different MFCC Features - Mean')
# Place a legend to the right of this smaller figure.
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.savefig('ex5_2_means.jpg')

i = 0
plt.figure(2)

while i < n:
	line1, = plt.plot(ax, varis[i], color[i] + '-', label=names[i])
	plt.hold(True)
	i = i + 1

plt.title('VARIANCES')
plt.ylabel('Phoneme')
plt.xlabel('Different MFCC Features - Variance')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.savefig('ex5_2_vars.jpg')

i = 0
plt.figure(3)

while i < n:
	line1, = plt.plot(ax, idctmeans[i], color[i] + '-', label=names[i])
	plt.hold(True)
	i = i + 1

plt.title('INVERSE DCT')
plt.ylabel('Phoneme')
plt.xlabel('IDCT of MFCC Features')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.savefig('ex5_2_idct.jpg')


