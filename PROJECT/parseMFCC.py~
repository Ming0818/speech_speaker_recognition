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
labels = np.zeros((10000000,1))

ind = 0

with open(fl) as f:
	for line in f:
		if ind ==1000000:
			break;
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
		featVecs[ind,:] = fvparts
		labels[ind, :] = labelInt
		ind= ind+1

featVecs = featVecs[0:ind,:]
labels = labels[0:ind,:]


#print featVecs
#print labels

#exit()

featVecs = np.float32(featVecs)
labels = np.int_(labels)
labels = labels[:,0]

ind = 10000
print featVecs.shape
print labels.shape

featMean = np.mean(featVecs, axis=0)
featStd = np.std(featVecs, axis=0)

meanMat = np.tile(featMean.transpose(), (featVecs.shape[0], 1))
stdMat = np.tile(featStd.transpose(), (featVecs
.shape[0], 1))


featVecs = (featVecs - meanMat) / stdMat

trainInd = round(0.7*ind)
validInd = round(0.3*ind)
#testInd = round(0.1*ind)

feats1 = np.array([[0.2, 0.3, 0.5, 0.7, 0.9], [0.3, 0.4, 0.4, 0.8, 1.2], [0.3, 0.2, 0.5, 0.4, 0.8]])
labels1 = np.array([10, 5, 32])

#cPickle.dump([feats1, labels1], gzip.open('speechtr1.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)
#cPickle.dump([feats1, labels1], gzip.open('speechts1.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)
#exit()


featsTrain = featVecs[0:trainInd,:]
labelsTrain = labels[0:trainInd]
featsValid = featVecs[trainInd:trainInd+validInd, :]
labelsValid= labels[trainInd:trainInd+validInd]
#featsTest = featVecs[trainInd+validInd:trainInd+validInd+testInd,:]
#labelsTest = labels[trainInd+validInd:trainInd+validInd+testInd]
print featsTrain
print labelsTrain
#exit()


cPickle.dump([featsTrain, labelsTrain], gzip.open('speechtrain1.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)
cPickle.dump([featsValid, labelsValid], gzip.open('speechvalid1.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)
#cPickle.dump([featsTest, labelsTest], gzip.open('speechtest1.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)



		
		
