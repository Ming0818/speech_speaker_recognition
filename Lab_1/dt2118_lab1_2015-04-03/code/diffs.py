# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:59:22 2015

@author: thomai
"""

mat.subplot(141)
mat.imshow(mfcc_array[0])
mat.title(tidigits[0]['wid'])
mat.subplot(142)
mat.imshow(mfcc_array[1])
mat.title(tidigits[1]['wid'])
mat.subplot(143)
mat.imshow(mfcc_array[22])
mat.title(tidigits[22]['wid'])
mat.subplot(144)
mat.imshow(mfcc_array[23])
mat.title(tidigits[23]['wid'])
mat.show()