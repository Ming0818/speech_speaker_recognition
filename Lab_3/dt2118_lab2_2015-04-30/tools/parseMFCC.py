import os 
import sys
import cPickle, gzip
import numpy as np
import random

"""
file to read output file and convert it into pickle file.


"""
if len(sys.argv) < 2:
	sys.exit(0)


fl = sys.argv[1]
featVecs = np.zeros((10000000, 13))
labels = np.zeros((10000000, 1))

ind = 0

with open(fl) as f:
	for line in f:
		line = line.rstrip('\n')
		#print line		
		lp = line.split("&")
		#print lp
		featVec = lp[0]
		label = lp[1]

		fvparts = featVec.split(" ")
		fvparts.pop()
		fvparts = map(np.float32, fvparts)
		labelInt = int(label)
		#print len(fvparts)
		featVecs[ind, :] = fvparts
		labels[ind, :] = labelInt
		ind= ind+1

featVecs = featVecs[0:ind, :]
labels = labels[0:ind, :]


featVecs = np.float32(featVecs)
labels = np.int_(labels)
labels = labels[:, 0]


#trainInd = round(0.7 * ind)
#validInd = round(0.3 * ind)
testInd = round(0.3 * ind)


""" Normalize data """
featMean = np.mean(featVecs, axis=0)
featStd = np.std(featVecs, axis=0)

meanMat = np.tile(featMean.transpose(), (featVecs.shape[0], 1))
stdMat = np.tile(featStd.transpose(), (featVecs.shape[0], 1))


featVecs = (featVecs - meanMat) / stdMat



""" Assign features and labels to training, validation and testing """
featTest = featVecs[0:testInd, :]
labelsTest = labels[0:testInd]
#featsTrain = featVecs[0:trainInd, :]
#labelsTrain = labels[0:trainInd]
#featsValid = featVecs[trainInd:trainInd+validInd, :]
#labelsValid= labels[trainInd:trainInd + validInd]


#""" Normalize testing vector """
#featMean = np.mean(featTest, axis=0)
#featStd = np.std(featTest, axis=0)
#
#meanMat = np.tile(featMean.transpose(), (featTest.shape[0], 1))
#stdMat = np.tile(featStd.transpose(), (featTest.shape[0], 1))
#
#
#featTest = (featTest - meanMat) / stdMat
#
#
#""" Normalize training vector """
#featMeanTrain = np.mean(featsTrain, axis=0)
#featStdTrain = np.std(featsTrain, axis=0)
#
#meanMatTrain = np.tile(featMeanTrain.transpose(), (featsTrain.shape[0], 1))
#stdMatTrain = np.tile(featStdTrain.transpose(), (featsTrain.shape[0], 1))
#
#
#featTrain = (featsTrain - meanMatTrain) / stdMatTrain

#""" Normalize validation vector """
#featMeanValid = np.mean(featsValid, axis=0)
#featStdValid = np.std(featsValid, axis=0)
#
#meanMatValid = np.tile(featMeanValid.transpose(), (featsValid.shape[0], 1))
#stdMatValid = np.tile(featStdValid.transpose(), (featsValid.shape[0], 1))
#
#
#featsValid = (featsValid - meanMatValid) / stdMatValid

#cPickle.dump([featsTrain, labelsTrain], gzip.open('speechtrain1norm.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)
#cPickle.dump([featsValid, labelsValid], gzip.open('speechvalid1norm.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)
cPickle.dump([featTest, labelsTest], gzip.open('speechtest1norm.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)