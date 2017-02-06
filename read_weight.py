#!/usr/bin/env python

#import caffe
import caffe

import numpy as np

from matplotlib import pyplot

from compiler.ast import flatten

# added by yuzeng, used to plot the histogram for weights
def drawHist(weights):
    #the first parameter is the variable to be plot
    #the second parameter is the number of divided space
    pyplot.hist(weights, 100)
    pyplot.xlabel('weights')
    pyplot.ylabel('Frequency')
    pyplot.title('Weights of the network')
    pyplot.show()

# show the output param
np.set_printoptions(threshold='nan')

# deploy  file
# MODEL_FILE = 'caffe_deploy.prototxt'
MODEL_FILE = '/home/yuzeng/caffe/examples/mnist/lenet.prototxt'
# the trained caffe model
PRETRAIN_FILE = '/home/yuzeng/caffe/examples/mnist/lenet_iter_10000.caffemodel'

# file for storing the parameters
params_txt = 'params.txt'
pf = open(params_txt, 'w')

# read the paramters of caffe
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

# added by yuzeng, initialize a final weight array
weight_final=[]

# go over every layer
for param_name in net.params.keys():
    # weihts
    weight = net.params[param_name][0].data
    # biases
    bias = net.params[param_name][1].data



    # this layer corresponds the "top" lay in the prototxt 
    pf.write(param_name)
    pf.write('\n')

    # write the weights
    pf.write('\n' + param_name + '_weight:\n\n')
    # convert the multi-dimension to single-dimension
    weight.shape = (-1, 1)

    #drawHist(weight)

    for w in weight:
        pf.write('%f, ' % w)
        weight_final.append(float(w))

    # write biases
    pf.write('\n\n' + param_name + '_bias:\n\n')
    # convert the multi-dimension to single-dimension
    bias.shape = (-1, 1)
    for b in bias:
        pf.write('%f, ' % b)

    pf.write('\n\n')



drawHist(weight_final)

pf.close