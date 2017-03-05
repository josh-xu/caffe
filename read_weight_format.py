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
    pyplot.xlabel('Weights')
    pyplot.ylabel('Frequency')
    pyplot.title('Weights of the Network')
    pyplot.show()

def drawPlot(weights):
    pyplot.plot(weights)
    pyplot.xlabel('Index')
    pyplot.ylabel('Weights')
    pyplot.title('Weights of the Network')
    pyplot.show()

# show the output param
np.set_printoptions(threshold='nan')

# deploy file
# MODEL_FILE = 'caffe_deploy.prototxt'
MODEL_FILE = '../caffe/examples/mnist/lenet.prototxt'
#MODEL_FILE = '../caffe_github/examples/mnist/lenet.prototxt'
#MODEL_FILE = '../ristretto/examples/mnist/lenet_quantized.prototxt'
#MODEL_FILE = '../ristretto/examples/mnist/lenet_quantized_2n.prototxt'
# the trained caffe model
PRETRAIN_FILE = '../caffe/examples/mnist/lenet_iter_20000.caffemodel'
#PRETRAIN_FILE = '../caffe_github/examples/mnist/lenet_iter_10000.caffemodel'
#PRETRAIN_FILE = '../ristretto/examples/mnist/ristretto_lenet_iter_10000.caffemodel'
#PRETRAIN_FILE = '../ristretto/examples/mnist/ristretto_lenet_2n_iter_10000.caffemodel'

# file for storing the parameters
params_txt = 'params.txt'
pf = open(params_txt, 'w')

# read the paramters of caffe
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

print("the following is the data shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
[(k, v[0].data, v[1].data) for k, v in net.params.items()]

w1=net.params['conv1'][0].data
b1=net.params['conv1'][1].data
w2=net.params['conv2'][0].data
b2=net.params['conv2'][1].data
w3=net.params['ip1'][0].data
b3=net.params['ip1'][1].data
w4=net.params['ip2'][0].data
b4=net.params['ip2'][1].data

print "conv1 weights"
print net.params['conv1'][0].data.shape
#print net.params['pool1'][0].data.shape
print "conv2 weights"
print net.params['conv2'][0].data.shape
#print net.params['pool2'][0].data.shape
print "ip1 weights"
print net.params['ip1'][0].data.shape
print "ip2 weights"
print net.params['ip2'][0].data.shape
#print net.params['ip1'][0].data[1][1]
#print net.params['ip1'][0].data[0]
#print net.params['ip1'][0].data[0].shape
FD=file('/home/yuzeng/caffe/weight.txt','w+')
net.forward()
"""
print "data"
print net.blobs['data'].data.shape
print "conv1"
print net.blobs['conv1'].data.shape
print >> FD, "conv1"
print >> FD, net.blobs['conv1'].data
print "pool1"
print net.blobs['pool1'].data.shape
print >> FD, "pool1"
print >> FD, net.blobs['pool1'].data
print "conv2"
print net.blobs['conv2'].data.shape
print >> FD, "conv2"
print >> FD, net.blobs['conv2'].data
print "pool2"
print net.blobs['pool2'].data.shape
print >> FD, "pool2"
print >> FD, net.blobs['pool2'].data
print "ip1"
print net.blobs['ip1'].data.shape
print >> FD, "ip1"
print >> FD, net.blobs['ip1'].data
print "ip2"
print net.blobs['ip2'].data.shape
print >> FD, "ip2"
print >> FD, net.blobs['ip2'].data
"""

# added by yuzeng, initialize a final weight array
#weight_final=[]

# go over every layer
"""for param_name in net.params.keys():
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
        pf.write('%.10f, ' % w)
        #pf.write('%s, ' % float(w))
        weight_final.append(float(w))

    # write biases
    pf.write('\n\n' + param_name + '_bias:\n\n')
    # convert the multi-dimension to single-dimension
    bias.shape = (-1, 1)
    for b in bias:
        pf.write('%f, ' % b)

    pf.write('\n\n')

#drawHist(weight_final)
weight_final.sort()
#drawPlot(weight_final)
"""
pf.close
