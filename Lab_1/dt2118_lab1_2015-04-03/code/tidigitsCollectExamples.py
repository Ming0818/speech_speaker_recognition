# Script to collect examples from the TIDIGITS database
# To be able to run you need:
# - scikits.audiolab (https://pypi.python.org/pypi/scikits.audiolab/)
# - access to KTH afs cell nada.kth.se
# - access rights to the TIDIGITS database
#
# Usage:
# python tidigitsCollectExamples.py
#
# (C) 2015 Giampiero Salvi <giampi@kth.se>
# DT2118 Speech and Speaker Recognition
import numpy as np
import os
from scikits.audiolab import Sndfile

tidigitsroot = '/afs/nada.kth.se/dept/tmh/corpora/tidigits/disc_4.1.1/tidigits/train/'

genders = ["man", "woman"]
speakers = ["ae", "ac"]

digits = ["o", "z", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
repetitions = ["a", "b"]

tidigits = []
for idx in range(len(speakers)):
    for digit in digits:
        for repetition in repetitions:
            filename = os.path.join(tidigitsroot, genders[idx], speakers[idx], digit+repetition+'.wav')
            sndobj = Sndfile(filename)
            samples = np.array(sndobj.read_frames(sndobj.nframes))
            samplingrate = sndobj.samplerate
            tidigits.append({"filename": filename, "samplingrate": samplingrate, "gender": genders[idx], "speaker": speakers[idx], "digit": digit, "repetition": repetition, "samples": samples})

np.savez('tidigits_examples.npz', tidigits=tidigits)
