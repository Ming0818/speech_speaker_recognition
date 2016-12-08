import numpy as np
from tools2 import *


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def my_range_neg(start, end, step):
    while start >= end:
        yield start
        start -= step



def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
        N observations(frames), K models
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    N_K = log_emlik.shape;
    like = 0;
    for i in range(N_K[0]):
        like += logsumexp(log_emlik[i, :] + np.log(weights));

    return like;


def forward(log_emlik, log_startprob, log_transmat):
    """Forward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames(=timesteps), M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j
        N = number of emissions
        M = number of states
    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    states = log_transmat.shape[0];
    emissions = log_emlik.shape[0];

    forward_prob = np.zeros((emissions, states));
    """
        Initialize alphas
    """
    for j in range(states):
        forward_prob[0, j] = log_startprob[j] + log_emlik[0, j];

    for i in my_range(1, emissions - 1, 1):
        for j in range(states):
            temp = np.transpose(forward_prob[i - 1, :]) + log_transmat[:, j];
            forward_prob[i, j] = logsumexp(temp) + log_emlik[i, j];
    return forward_prob



def backward(log_emlik, log_startprob, log_transmat):
    """Backward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames(=time steps), M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    states = log_transmat.shape[0];
    emissions = log_emlik.shape[0];

    V = np.zeros((emissions, states));
    B = np.zeros((emissions, states));
    viterbi_path = np.zeros(emissions);
    """
        Initialize
    """
    for j in range(states):
        V[0, j] = log_startprob[j] + log_emlik[0, j];

    for i in my_range(1, emissions - 1, 1):
        for j in range(states):
            ''' Find max '''
            temp = V[i - 1, :] + np.transpose(log_transmat[:, j]);
            V[i, j] = np.max(temp) + log_emlik[i, j];
            B[i, j] = np.argmax(temp);

    viterbi_path[emissions - 1] = np.argmax(V[emissions - 1, :]);

    """ Backtracking """
    for i in my_range_neg(emissions - 2, 0, 1):
        viterbi_path[i] = B[i + 1, viterbi_path[i + 1]];

    return np.max(V[emissions - 1, :]), viterbi_path;