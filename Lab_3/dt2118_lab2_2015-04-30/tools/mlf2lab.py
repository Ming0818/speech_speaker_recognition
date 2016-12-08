import sys
import numpy as np
import htkmfc

"""
This code is for converting HTK feature and label files into the PFile format. The label file looks like
-------------------------------------------------------------
#!MLF!#
"/data/utt1.lab"
0 400000 sil[2] -246.995255 sil
400000 1000000 sil[3] -350.304596
1000000 1100000 sil[4] -69.883293
1100000 1300000 sil-k+ih[2] -147.455048 sil-k+ih
1300000 1400000 sil-k+ih[3] -85.606064
1400000 1600000 sil-k+ih[4] -183.568436
.
"/data/utt2.lab"
0 100000 sil[2] -44.788208 sil
100000 600000 sil[3] -238.076736
600000 700000 sil[4] -58.250130
700000 1700000 sil-hh+ih[2] -699.987000 sil-hh+iy
1700000 2000000 sil-hh+ih[3] -222.936462
2000000 2200000 sil-hh+ih[4] -166.201675
...
------------------------------------------------------------

Please note:
1. htkmfc.py should be downloaded and put under the same directory
http://sphinx-am2wfst.googlecode.com/hg-history/c08d2c86a25c491116bb70b729fb7f2cda7bda2e/t3sphinx/htkmfc.py

2. Feature files have to be under the same directory as label files, and with the extension ".mfc". That is, the feature file corresponding to /data/utt1.lab should be /data/utt1.mfc

3. pfile_utils has been installed and added to the environmental path. It can be downloaded from here
http://www.icsi.berkeley.edu/ftp/pub/real/davidj/pfile_utils-v0_51.tar.gz

Also, running the following script will install it automatically
http://www.cs.cmu.edu/~ymiao/kaldipdnn/install_pfile_utils.sh

"""

usage = """
python htk_to_pfile.py mlf_filename state_str2int | pfile_create -i - -o sample.pfile -f $feat_dim -l 1
e.g., python mlf2lab.py train_aligned.mlf phoneme2num.txt > train_data.txt

The file state_str2int contains mapping for states from the string format to integers, such as
sil-k+ih[2]	1
sil-k+ih[3]	2
sil-k+ih[4]	3
ih-n+l[2]	4

"""

# Configuration
MFCC_TIMESTEP = 10    # the frame shift in terms of ms

if __name__ == '__main__':
    folder = '.'
    if len(sys.argv) < 2:
        print usage
        sys.exit(0)
    mlf = sys.argv[1]
   # print "Producing a (x, y) dataset file for:", mlf
    
    # reading states mapping: string --> integer
    str2int = {}
    with open(sys.argv[2]) as fin:
        for line in fin:
            lp = line.strip().split("\t")
            if len(lp) > 1:
            	str2int[lp[0]] = int(lp[1])

	
    with open(mlf) as f:
        tmp_len_x = 0 # verify sizes
        frame_num = 0 # for the current utterance, the number of frames
        feat_dim = 0  # the feature dimension
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 1:
                continue
            if line[0] == '"': # a new utterance
                # the corresponding feature file with the extension of .mfc
                mfc_file = line.strip('"')[:-3] + 'mfc'
	       # print "processing", mfc_file
		mfc_reader = htkmfc.open(mfc_file)   # .lab -> .mfc
                x = mfc_reader.getall()
                frame_num, feat_dim = x.shape
            elif line[0].isdigit():
                start, end, state = line.split()[:3]
                # start and end frame index
                start = (int(start)+1)/(MFCC_TIMESTEP * 10000)
                end = (int(end)+1)/(MFCC_TIMESTEP * 10000)
				
                for i in xrange(start, end):
                    line = ''
                    for j in xrange(feat_dim):
                        line = line + str(x[i, j]) + ' '
		    line = line + '&' +str(str2int[state])    # append the state label for each frame
                    print line    # print each line to the standard output, then pipe to pfile_create


