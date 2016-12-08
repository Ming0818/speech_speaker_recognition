#!/bin/bash

# two variables you need to set
pdnndir=/home/thomai/pdnn  # pointer to PDNN
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32


# train DNN model
echo "Training the DNN model ..."
python $pdnndir/cmds/run_DNN.py --train-data "speechtrain1norm.pickle.gz" \
                                --valid-data "speechvalid1norm.pickle.gz" \
                                --nnet-spec "13:256:256:256:256:22" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.2:100" --model-save-step 20 \
                                --param-output-file dnn256_256_256_256_0.2_100_norm.param \
                                --cfg-output-file dnn256_256_256_256_0.2_100_norm.cfg  >& dnn.training256_256_256_256_0.2_100_norm.log

# classification on the testing data; -1 means the final layer, that is, the classification softmax layer
echo "Classifying with the DNN model ..."
python $pdnndir/cmds/run_Extract_Feats.py --data "speechtest1norm.pickle.gz" \
                                          --nnet-param dnn256_256_256_256_0.2_100_norm.param --nnet-cfg dnn256_256_256_256_0.2_100_norm.cfg \
                                          --output-file "dnn256_256_256_256_0.2_100_norm.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >& dnn.testing256_256_256_256_0.2_100_norm.log

python show_results.py dnn256_256_256_256_0.2_100_norm.classify.pickle.gz > result256_256_256_256_0.2_100_norm.txt

# train CNN model
#echo "Training the CNN model ..."
#python $pdnndir/cmds/run_CNN.py --train-data "train.pickle.gz" \
#                                --valid-data "valid.pickle.gz" \
#                                --conv-nnet-spec "1x28x28:20,5x5,p2x2:50,5x5,p2x2,f" --nnet-spec "512:10" --wdir ./ \
#                                --l2-reg 0.0001 --lrate "C:0.1:200" --model-save-step 20 \
#                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >& cnn.training.log

#echo "Classifying with the CNN model ..."
#python $pdnndir/cmds/run_Extract_Feats.py --data "test.pickle.gz" \
#                                          --nnet-param cnn.param --nnet-cfg cnn.cfg \
#                                          --output-file "cnn.classify.pickle.gz" --layer-index -1 \
#                                          --batch-size 100 >& cnn.testing.log

#python show_results.py cnn.classify.pickle.gz
