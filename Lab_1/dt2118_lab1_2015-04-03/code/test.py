# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:33:50 2015

@author: thomai
"""

import numpy as np
import matplotlib.pyplot as mat
import scipy.signal
import scipy.fftpack
from tools import trfbank
tidigits = np.load('tidigits_examples.npz')['tidigits']
example = np.load('tidigits_examples.npz')['example']


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def enframe(samples, winlen, winshift):
    """Slices the input samples into overlapping windows.

    Args:
        winlen: window lenght in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    samples_len = len(samples);
    if (samples_len <= winlen):
        return samples;
    windows = np.array;
    windows = samples[0 : winlen];
    for i in my_range(winshift, samples_len - winlen, winshift):
        windows = np.vstack([windows, samples[i:i + winlen]]);
    return windows;

def preemp(input, p=0.97):
    """Pre-emphasis filter.

    Args:
        input: array of speech samples
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of filtered speech samples
    """
    size = input.shape;
    a = np.zeros(size[1]);
    a[0] = 1;
    b = np.zeros(size[1]);
    b[0] = 1;
    b[1] = -p;
    return scipy.signal.lfilter(b, a, input, axis=1);


"""
    Begin main
"""
for idx in range(len(tidigits)):
    tidigits[idx]['wid'] = tidigits[idx]['gender'] + '_' + \
      tidigits[idx]['speaker'] + '_' + tidigits[idx]['digit'] + \
      tidigits[idx]['repetition'];
# Set window length and window shift in samples
winlen = 400  # for window length 20ms
winshift = 200  # for window shift 10ms

# Get frames
frames = enframe(example[0]['samples'], winlen, winshift);
size = frames.shape;
# Pre-emphasize
emphasized = preemp(frames);
# Define and apply hamming window
ham = scipy.signal.hamming(winlen, sym=False);
#mat.plot(ham);
#mat.show();
windowed = np.zeros(size);
for i in range(size[0]):
    windowed[i, :] = [a*b for a,b in zip(ham, emphasized[i, :])];


# Apply fft
spec = np.zeros((size[0], 512));
logspec = np.zeros((size[0], 512));
for i in range(size[0]):
    spec[i, :] = np.abs(scipy.fftpack.fft(windowed[i, :], 512));
spec = np.power(spec, 2);
logspec = np.log10(spec);

# Create bank of triangular filters
filters = trfbank(20000, 512);
filter_size = filters.shape;
#for i in range(filter_size[0]):
#    mat.plot(filters[i, :]);
#mat.show();
mspec = np.dot(spec, np.transpose(filters));
mspec = np.log10(mspec);

# Cosine transform
mfcc = np.zeros((size[0], 13));
temp = scipy.fftpack.realtransforms.dct(mspec, type=2, axis = 1, norm='ortho');
mfcc = temp[:, 0:13];


mat.subplot(211)
mat.imshow(np.transpose(example[0]['mfcc']))
mat.title('expected')
mat.subplot(212)
mat.imshow(np.transpose(mfcc))
mat.title('mine')
mat.show()