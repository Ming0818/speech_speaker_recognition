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
from matplotlib.legend_handler import HandlerLine2D

hmmdir = sys.argv[1]
i = 0
n_hmm = 8
means= np.zeros((n_hmm,13))
varis= np.zeros((n_hmm,13))
mel = ['coeff 0','coeff 1','coeff 2','coeff 3','coeff 4','coeff 5','coeff 6','coeff 7','coeff 8','coeff 9','coeff 10','coeff 11','coeff 12']
while i < n_hmm:
    hmmfile = hmmdir + '/hmm' + str(i) + '/hmmdefs.mmf'
    #print('Example usage for Georgi Dzhambazov\'s HTK model parser')
    print('opening HTK model file: '+ hmmfile)
    file = open(hmmfile)
    #print('parsing the data')
    data = file.read()
    hmms = yacc.parse(data)
    #print('the number of HMM models is: ' + str(len(hmms)))
    m=10
    thishmm = hmms[m]
    print('model number '+str(m+1)+' is called '+thishmm.name+' and has '+str(len(thishmm.states))+' states')
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
    means[i] = thismean
    varis[i] = thisvar
    i = i + 1

##Plot

ax = np.linspace(1,n_hmm ,n_hmm)

i = 0
color = ['.r','.b','.g','.y','.k','or','ob','og','oy','ok','*r','*b','*g']
plt.figure(1)

while i < 13:
	line1, = plt.plot(ax, means.T[i], color[i] + '-', label=mel[i])
	plt.hold(True)
	i = i + 1

plt.title('MEANS')
plt.xlabel('HMM Models')
plt.ylabel('Different MFCC Features - Mean')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.savefig('ex5_1_means.jpg')

i = 0
plt.figure(2)

while i < 13:
	line1, = plt.plot(ax, varis.T[i], color[i] + '-', label=mel[i])
	plt.hold(True)
	i = i + 1

plt.title('VARIANCES')
plt.xlabel('HMM Models')
plt.ylabel('Different MFCC Features - Variance')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.savefig('ex5_1_vars.jpg')


