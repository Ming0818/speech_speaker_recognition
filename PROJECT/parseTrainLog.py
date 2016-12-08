import os
import re
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

contents = os.listdir('.');
#filename = 'dnn.training128_128_0.3_100.log'

ve = {}
te = {}
epochs = np.arange(10) + 1

tetext='training error'
vetext='validation error'

for i in range(len(contents)):
    if (os.path.isdir(contents[i]) and contents[i] != '.git'):
        with open(contents[i] + '/dnn.training' + contents[i] + '.log') as fl:
            temp_te = []
            temp_ve = []
            for line in fl:
                line = line.strip('\n')
                # print line
                lp1 = line.split(' ')
                if tetext in line:
                    epoch = int(lp1[4].rstrip(','))
                    trval = float(lp1[7])
                    temp_te.append(trval)
                te[contents[i]] = temp_te
                
                if vetext in line:
                    veval = float(lp1[9])
                    temp_ve.append(veval)
                ve[contents[i]] = temp_ve


non_norm_train = plt.figure()
non_norm_valid = plt.figure()
norm_train = plt.figure()
norm_valid = plt.figure()


for i in range(len(contents)):
    if (os.path.isdir(contents[i]) and contents[i] != '.git'):
        if (contents[i][-1] == 'm'):
            # Normalized data
            #Plot training error
            plt.figure(norm_train.number)
            line1, = plt.plot(range(100), te[contents[i]], label=contents[i])
            #Plot validation error
            plt.figure(norm_valid.number)
            line2, = plt.plot(range(100), ve[contents[i]], label=contents[i])
        else:
            # Non - normalized data
            #Plot training error
            plt.figure(non_norm_train.number)
            line3, = plt.plot(range(100), te[contents[i]], label=contents[i])
            #Plot validation error
            plt.figure(non_norm_valid.number)
            line4, = plt.plot(range(100), ve[contents[i]], label=contents[i])
            

plt.figure(norm_train.number)
plt.title('Training error for normalized data')
plt.xlabel('Epochs')
plt.ylabel('Training error')
plt.grid()
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

plt.figure(norm_valid.number)
plt.title('Validation error for normalized data')
plt.xlabel('Epochs')
plt.ylabel('Validation error')
plt.grid()
plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})

plt.figure(non_norm_train.number)
plt.title('Training error for non-normalized data')
plt.xlabel('Epochs')
plt.ylabel('Training error')
plt.grid()
plt.legend(handler_map={line3: HandlerLine2D(numpoints=4)})

plt.figure(non_norm_valid.number)
plt.title('Validation error for non-normalized data')
plt.xlabel('Epochs')
plt.ylabel('Validation error')
plt.grid()
plt.legend(handler_map={line4: HandlerLine2D(numpoints=4)})
     
        
        
#fig1 = plt.figure(1)
#ax1 = fig1.add_subplot(111)
#ax1.plot(epochs, te, 'r--', epochs, ve)
#
#plt.axis([0, 100, 0, 100])
#axes_position = [0, 100, 0, 100]
#
#
#ax = fig1.add_axes(axes_position)
#
## plt.show()
#plt.savefig('error-128-0.3.png')


