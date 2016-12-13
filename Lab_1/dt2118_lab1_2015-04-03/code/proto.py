# DT2118, Lab 1 Feature Extraction
# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """Slices the input samples into overlapping windows.

    Args:
        winlen: window lenght in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    
def preemp(input, p=0.97):
    """Pre-emphasis filter.

    Args:
        input: array of speech samples
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of filtered speech samples
    """

def mfcc(samples, winlen, winshift, nfft, nceps, samplingrate):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        nfft: length of the Fast Fourier Transform (power of 2, grater than winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal

    Note: for convenienve, you can define defaults for the input arguments that fit the exercise

    Returns:
        ceps: N x nceps array with one MFCC feature vector per row
        mspec: N x M array of outputs of the Mel filterbank (of size M)
        spec: N x nfft array with squared absolute fast Fourier transform
    """

def dtw(localdist):
    """Dynamic Time Warping.

    Args:
        localdist: array NxM of local distances computed between two sequences
                   of length N and M respectively

    Output:
        globaldist: scalar, global distance computed by Dynamic Time Warping
    """
