from scipy.io import savemat
import numpy as np

tidigits_examples = np.load('tidigits_examples.npz')
savemat('tidigits_examples.mat', tidigits_examples)
