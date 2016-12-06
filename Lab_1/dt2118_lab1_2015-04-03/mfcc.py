# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:42:56 2015

@author: thomai
"""


import numpy as np
import matplotlib.pyplot as mat
import scipy.signal
import scipy.fftpack
from tools import trfbank
from sklearn.mixture import GMM
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


def mfcc(samples, winlen=400, winshift=200, nfft=512, nceps=13, samplingrate=20000):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: length of the analysis window
        winshift: number of samples to shift the analysis window at every time
            step
        nfft: length of the Fast Fourier Transform (power of 2, grater than
            winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal

    Note: for convenienve, you can define defaults for the input arguments
        that fit the exercise

    Returns:
        ceps: N x nceps array with one MFCC feature vector per row
        mspec: N x M array of outputs of the Mel filterbank (of size M)
        spec: N x nfft array with squared absolute fast Fourier transform
    """
    # Get frames
    frames = enframe(samples, winlen, winshift);

    size = frames.shape;

    # Pre-emphasize
    emphasized = preemp(frames);

    # Define and apply hamming window
    ham = scipy.signal.hamming(winlen, sym=False);
    windowed = np.zeros(size);
    for i in range(size[0]):
        windowed[i, :] = [a*b for a,b in zip(ham, emphasized[i, :])];

    # Apply fft
    spec = np.zeros((size[0], nfft));
    logspec = np.zeros((size[0], nfft));
    for i in range(size[0]):
        spec[i, :] = np.abs(scipy.fftpack.fft(windowed[i, :], nfft));
    spec = np.power(spec, 2);
    logspec = np.log10(spec);



    # Create bank of triangular filters
    filters = trfbank(samplingrate, nfft);
    filter_size = filters.shape;
#    for i in range(filter_size[0]):
#        mat.plot(filters[i, :]);
#    mat.show();
    mspec = np.dot(spec, np.transpose(filters));
    mspec = np.log10(mspec);

    # Cosine transform
    mfcc = np.zeros((size[0], nceps));
    temp = scipy.fftpack.realtransforms.dct(mspec, type=2, axis=1, norm='ortho');
    mfcc = temp[:, 0:nceps];

    return mfcc, mspec, spec;

def dtw(localdist):
    """Dynamic Time Warping.
    Args:
        localdist: array NxM of local distances computed between two sequences
                   of length N and M respectively
    Output:
        globaldist: scalar, global distance computed by Dynamic Time Warping
    """
    size = localdist.shape;
    glob = np.zeros(size);
    for i in range(size[0]):
        for j in range(size[1]):
            glob[i, j] = localdist[i, j] + np.min([glob[i - 1, j], \
              glob[i - 1, j - 1], glob[i, j - 1]]);
    return glob[-1, -1]


"""
    MAIN
"""
for idx in range(len(tidigits)):
    tidigits[idx]['wid'] = tidigits[idx]['gender'] + '_' + \
      tidigits[idx]['speaker'] + '_' + tidigits[idx]['digit'] + \
      tidigits[idx]['repetition'];


#mfcc(example[0]['samples']);


utterances = len(tidigits);
features = np.array;
mspecs = np.array;
mfcc_array = {};
mspec = {};
spec = {};
for i in range(len(tidigits)):
#    print tidigits[i]['wid']
    (temp1, temp2, temp3) = mfcc(tidigits[i]['samples']);
    mfcc_array[i] = temp1;
    mspec[i] = temp2;
    spec[i] = temp3;
    if i == 0:
        features = temp1;
        mspecs = temp2;
    else:
        features = np.vstack([features, temp1]);
        mspecs = np.vstack([mspecs, temp2]);

print "Calculated features"
#feat_size = features.shape;
#
## Create gaussian mixture model
#gmm = GMM(n_components=16, covariance_type='diag');
#gmm.fit(features);
#print 'Trained Gaussian model'

#'''
#    Standarize MFCCs: remove global mean and divide by global standard deviation
#'''
#m = np.mean(features);
#s = np.std(features);
#mfcc_array_std = {};
#for i in range(len(mfcc_array)):
#    mfcc_array_std[i] = (mfcc_array[i] - m) / s;

#probs = {};
#for i in range(len(tidigits)):
#    probs[i] = gmm.predict_proba(mfcc_array[i]);
#
#
## Calculate local distances
#D_probs = np.zeros((utterances, utterances));
#for utt1 in range(utterances):
#    print 'utt1: ', utt1;
#    size_utt1 = probs[utt1].shape;
#    for utt2 in range(utterances):
#        size_utt2 = probs[utt2].shape;
#        # Create local distance matrix
#        local_dist = np.zeros((size_utt1[0], size_utt2[0]));
#        for i in range(size_utt1[0]):
#            a = probs[utt1][i, :];
#            for j in range(size_utt2[0]):
#                b = probs[utt2][j, :];
#                local_dist[i, j] = np.linalg.norm(a-b);
#        # We have local distances for the two utterances
#        # Calculate local distances of utterances and store in D
#        D_probs[utt1, utt2] = dtw(local_dist);
#
#mat.imshow(D_probs);
#mat.show();



## Calculate correlations
#corr = np.corrcoef(mspecs, rowvar=0);

