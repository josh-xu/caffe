#!/usr/bin/env python

#import caffe
import caffe

import math

import numpy as np

from matplotlib import pyplot

from compiler.ast import flatten

'''
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
'''

# show the output param
np.set_printoptions(threshold='nan')

# deploy file
# MODEL_FILE = 'caffe_deploy.prototxt'
MODEL_FILE = '../caffe/examples/mnist/lenet.prototxt'
#MODEL_FILE = '../caffe_github/examples/mnist/lenet.prototxt'
#MODEL_FILE = '../ristretto/examples/mnist/lenet_quantized.prototxt'
#MODEL_FILE = '../ristretto/examples/mnist/lenet_quantized_2n.prototxt'
# the trained caffe model
PRETRAIN_FILE = '../caffe/examples/mnist/lenet_iter_30000.caffemodel'
#PRETRAIN_FILE = '../caffe_github/examples/mnist/lenet_iter_10000.caffemodel'
#PRETRAIN_FILE = '../ristretto/examples/mnist/ristretto_lenet_iter_10000.caffemodel'
#PRETRAIN_FILE = '../ristretto/examples/mnist/ristretto_lenet_2n_iter_10000.caffemodel'

# file for storing the parameters
params_txt = 'params.txt'
pf = open(params_txt, 'w')

# read the paramters of caffe
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

print("@@@the following is the data shape!")
#[(k, v[0].data, v[1].data) for k, v in net.params.items()]

#w1=net.params['conv1'][0].data
#b1=net.params['conv1'][1].data
#w2=net.params['conv2'][0].data
#b2=net.params['conv2'][1].data
#w3=net.params['ip1'][0].data
#b3=net.params['ip1'][1].data
#w4=net.params['ip2'][0].data
#b4=net.params['ip2'][1].data

#conv
N=0
K=0
H=0
W=0

#ip
N=0
D=0

layer_num = 0

conv1_centroids=[]
conv2_centroids=[]
ip1_centroids=[]
ip2_centroids=[]
with open('../caffe/train_model_paras_centroids.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_centroids.append(float(data[0]))
        elif layer_num == 2:
            conv2_centroids.append(float(data[0]))
        elif layer_num == 3:
            ip1_centroids.append(float(data[0]))
        elif layer_num == 4:
            ip2_centroids.append(float(data[0]))

conv1_indices=[]
conv2_indices=[]
ip1_indices=[]
ip2_indices=[]
with open('../caffe/train_model_paras_indices.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_indices.append(int(data[0]))
        elif layer_num == 2:
            conv2_indices.append(int(data[0]))
        elif layer_num == 3:
            ip1_indices.append(int(data[0]))
        elif layer_num == 4:
            ip2_indices.append(int(data[0]))

conv1_masks=[]
conv2_masks=[]
ip1_masks=[]
ip2_masks=[]
with open('../caffe/train_model_paras_masks.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks.append(float(data[0]))

conv1_masks1=[]
conv2_masks1=[]
ip1_masks1=[]
ip2_masks1=[]
with open('../caffe/train_model_paras_masks1.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks1.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks1.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks1.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks1.append(float(data[0]))

conv1_masks2=[]
conv2_masks2=[]
ip1_masks2=[]
ip2_masks2=[]
with open('../caffe/train_model_paras_masks2.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks2.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks2.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks2.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks2.append(float(data[0]))

conv1_masks3=[]
conv2_masks3=[]
ip1_masks3=[]
ip2_masks3=[]
with open('../caffe/train_model_paras_masks3.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks3.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks3.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks3.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks3.append(float(data[0]))

conv1_masks4=[]
conv2_masks4=[]
ip1_masks4=[]
ip2_masks4=[]
with open('../caffe/train_model_paras_masks4.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks4.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks4.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks4.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks4.append(float(data[0]))

conv1_masks5=[]
conv2_masks5=[]
ip1_masks5=[]
ip2_masks5=[]
with open('../caffe/train_model_paras_masks5.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks5.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks5.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks5.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks5.append(float(data[0]))

conv1_masks5p6=[]
conv2_masks5p6=[]
ip1_masks5p6=[]
ip2_masks5p6=[]
with open('../caffe/train_model_paras_masks5p6.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks5p6.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks5p6.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks5p6.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks5p6.append(float(data[0]))

conv1_masks5p7=[]
conv2_masks5p7=[]
ip1_masks5p7=[]
ip2_masks5p7=[]
with open('../caffe/train_model_paras_masks5p7.log') as f:
    for line in f:
        data = line.split()
        if data[0] == '@@@conv1':
            layer_num = 1
            continue
        elif data[0] == '@@@conv2':
            layer_num = 2
            continue
        elif data[0] == '@@@ip1':
            layer_num = 3
            continue
        elif data[0] == '@@@ip2':
            layer_num = 4
            continue

        if layer_num == 1:
            conv1_masks5p7.append(float(data[0]))
        elif layer_num == 2:
            conv2_masks5p7.append(float(data[0]))
        elif layer_num == 3:
            ip1_masks5p7.append(float(data[0]))
        elif layer_num == 4:
            ip2_masks5p7.append(float(data[0]))

CONV_QUNUM = 256
FC_QUNUM = 32
CONV1_COUNT = 500
CONV2_COUNT = 25000
IP1_COUNT = 400000
IP2_COUNT = 5000

# something wired with count of masks_all.... don't know why!!!
conv1_masks_all=[0] * CONV1_COUNT
conv2_masks_all=[0] * CONV2_COUNT
ip1_masks_all=[0] * IP1_COUNT
ip2_masks_all=[0] * IP2_COUNT
for i in range(0, CONV1_COUNT):
    conv1_masks_all[i] = conv1_masks[i] and conv1_masks1[i]
    conv1_masks_all[i] = conv1_masks_all[i] and conv1_masks2[i] and conv1_masks3[i] and conv1_masks4[i] and conv1_masks5[i] and conv1_masks5p6[i] and conv1_masks5p7[i]
for i in range(0, CONV2_COUNT):
    conv2_masks_all[i] = conv2_masks[i] and conv2_masks1[i]
    conv2_masks_all[i] = conv2_masks_all[i] and conv2_masks2[i] and conv2_masks3[i] and conv2_masks4[i] and conv2_masks5[i] and conv2_masks5p6[i] and conv2_masks5p7[i]
for i in range(0, IP1_COUNT):
    ip1_masks_all[i] = ip1_masks[i] and ip1_masks1[i]
    ip1_masks_all[i] = ip1_masks_all[i] and ip1_masks2[i] and ip1_masks3[i] and ip1_masks4[i] and ip1_masks5[i] and ip1_masks5p6[i] and ip1_masks5p7[i]
for i in range(0, IP2_COUNT):
    ip2_masks_all[i] = ip2_masks[i] and ip2_masks1[i]
    ip2_masks_all[i] = ip2_masks_all[i] and ip2_masks2[i] and ip2_masks3[i] and ip2_masks4[i] and ip2_masks5[i] and ip2_masks5p6[i] and ip2_masks5p7[i]

if len(conv1_centroids) != CONV_QUNUM:
    print "error!"
if len(conv1_indices) != CONV1_COUNT:
    print "error!"
if len(conv1_masks_all) != CONV1_COUNT:
    print "error!"
if len(conv1_masks) != CONV1_COUNT:
    print "error!"
if len(conv1_masks1) != CONV1_COUNT:
    print "error!"
if len(conv1_masks2) != CONV1_COUNT:
    print "error!"
if len(conv1_masks3) != CONV1_COUNT:
    print "error!"
if len(conv1_masks4) != CONV1_COUNT:
    print "error!"
if len(conv1_masks5) != CONV1_COUNT:
    print "error!"
if len(conv1_masks5p6) != CONV1_COUNT:
    print "error!"
if len(conv1_masks5p7) != CONV1_COUNT:
    print "error!"

if len(conv2_centroids) != CONV_QUNUM:
    print "error!"
if len(conv2_indices) != CONV2_COUNT:
    print "error!"
if len(conv2_masks_all) != CONV2_COUNT:
    print "error!"
if len(conv2_masks) != CONV2_COUNT:
    print "error!"
if len(conv2_masks1) != CONV2_COUNT:
    print "error!"
if len(conv2_masks2) != CONV2_COUNT:
    print "error!"
if len(conv2_masks3) != CONV2_COUNT:
    print "error!"
if len(conv2_masks4) != CONV2_COUNT:
    print "error!"
if len(conv2_masks5) != CONV2_COUNT:
    print "error!"
if len(conv2_masks5p6) != CONV2_COUNT:
    print "error!"
if len(conv2_masks5p7) != CONV2_COUNT:
    print "error!"

if len(ip1_centroids) != FC_QUNUM:
    print "error!"
if len(ip1_indices) != IP1_COUNT:
    print "error!"
if len(ip1_masks_all) != IP1_COUNT:
    print "error!"
if len(ip1_masks) != IP1_COUNT:
    print "error!"
if len(ip1_masks1) != IP1_COUNT:
    print "error!"
if len(ip1_masks2) != IP1_COUNT:
    print "error!"
if len(ip1_masks3) != IP1_COUNT:
    print "error!"
if len(ip1_masks4) != IP1_COUNT:
    print "error!"
if len(ip1_masks5) != IP1_COUNT:
    print "error!"
if len(ip1_masks5p6) != IP1_COUNT:
    print "error!"
if len(ip1_masks5p7) != IP1_COUNT:
    print "error!"

if len(ip2_centroids) != FC_QUNUM:
    print "error!"
if len(ip2_indices) != IP2_COUNT:
    print "error!"
if len(ip2_masks_all) != IP2_COUNT:
    print "error!"
if len(ip2_masks) != IP2_COUNT:
    print "error!"
if len(ip2_masks1) != IP2_COUNT:
    print "error!"
if len(ip2_masks2) != IP2_COUNT:
    print "error!"
if len(ip2_masks3) != IP2_COUNT:
    print "error!"
if len(ip2_masks4) != IP2_COUNT:
    print "error!"
if len(ip2_masks5) != IP2_COUNT:
    print "error!"
if len(ip2_masks5p6) != IP2_COUNT:
    print "error!"
if len(ip2_masks5p7) != IP2_COUNT:
    print "error!"

'''
print "conv1 weights"
print net.params['conv1'][0].data.shape
N = net.params['conv1'][0].data.shape[0]
K = net.params['conv1'][0].data.shape[1]
H = net.params['conv1'][0].data.shape[2]
W = net.params['conv1'][0].data.shape[3]

#print "pool1 weights"
#print net.params['pool1'][0].data.shape

print "conv2 weights"
print net.params['conv2'][0].data.shape

#print "pool2 weights"
#print net.params['pool2'][0].data.shape

print "ip1 weights"
print net.params['ip1'][0].data.shape

print "ip2 weights"
print net.params['ip2'][0].data.shape

#FD=file('../caffe/weight.txt','w+')
#net.forward()
'''

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
#"""
for param_name in net.params.keys():
    # weihts
    weight = net.params[param_name][0].data
    # biases
    bias = net.params[param_name][1].data

    print "%s weights" % param_name
    print weight.shape

    if param_name == 'conv1':
        N = weight.shape[0]
        K = weight.shape[1]
        H = weight.shape[2]
        W = weight.shape[3]
        for i in range(0, CONV1_COUNT):
            if conv1_masks_all[i]:
                net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = conv1_centroids[conv1_indices[i]]
            if not conv1_masks[i]:
                net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 0
            elif not conv1_masks1[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,1))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,1))
            elif not conv1_masks2[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,2))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,2))
            elif not conv1_masks3[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,3))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,3))
            elif not conv1_masks4[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,4))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,4))
            elif not conv1_masks5[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,5))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,5))
            elif not conv1_masks5p6[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,6))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,6))
            elif not conv1_masks5p7[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,7))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,7))
    elif param_name == 'conv2':
        N = weight.shape[0]
        K = weight.shape[1]
        H = weight.shape[2]
        W = weight.shape[3]
        for i in range(0, CONV2_COUNT):
            if conv2_masks_all[i]:
                net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = conv2_centroids[conv2_indices[i]]
            if not conv2_masks[i]:
                net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 0
            elif not conv2_masks1[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,1))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,1))
            elif not conv2_masks2[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,2))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,2))
            elif not conv2_masks3[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,3))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,3))
            elif not conv2_masks4[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,4))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,4))
            elif not conv2_masks5[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,5))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,5))
            elif not conv2_masks5p6[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,6))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,6))
            elif not conv2_masks5p7[i]:
                if net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] > 0:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,7))
                else:
                    net.params[param_name][0].data[(i/(K*H*W))%N][(i/(H*W))%K][(i/W)%H][i%W] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,7))
    elif param_name == 'ip1':
        N = weight.shape[0]
        D = weight.shape[1]
        for i in range(0, IP1_COUNT):
            if ip1_masks_all[i]:
                net.params[param_name][0].data[(i/D)%N][i%D] = ip1_centroids[ip1_indices[i]]
            if not ip1_masks[i]:
                net.params[param_name][0].data[(i/D)%N][i%D] = 0
            elif not ip1_masks1[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,1))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,1))
            elif not ip1_masks2[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,2))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,2))
            elif not ip1_masks3[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,3))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,3))
            elif not ip1_masks4[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,4))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,4))
            elif not ip1_masks5[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,5))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,5))
            elif not ip1_masks5p6[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,6))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,6))
            elif not ip1_masks5p7[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,7))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,7))
    elif param_name == 'ip2':
        N = weight.shape[0]
        D = weight.shape[1]
        for i in range(0, IP2_COUNT):
            if ip2_masks_all[i]:
                net.params[param_name][0].data[(i/D)%N][i%D] = ip2_centroids[ip2_indices[i]]
            if not ip2_masks[i]:
                net.params[param_name][0].data[(i/D)%N][i%D] = 0
            elif not ip2_masks1[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,1))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,1))
            elif not ip2_masks2[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,2))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,2))
            elif not ip2_masks3[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,3))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,3))
            elif not ip2_masks4[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,4))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,4))
            elif not ip2_masks5[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,5))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,5))
            elif not ip2_masks5p6[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,6))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,6))
            elif not ip2_masks5p7[i]:
                if net.params[param_name][0].data[(i/D)%N][i%D] > 0:
                    net.params[param_name][0].data[(i/D)%N][i%D] = 1.0/(math.pow(2,5)) + 1.0/(math.pow(2,7))
                else:
                    net.params[param_name][0].data[(i/D)%N][i%D] = -1.0/(math.pow(2,5)) - 1.0/(math.pow(2,7))

    #weight.shape = (-1, 1)

    '''
    # this layer corresponds the "top" lay in the prototxt 
    pf.write('##################### ' + param_name + ' #####################')
    pf.write('\n')

    # write the weights
    pf.write('\n' + param_name + '_weight:\n\n')
    # convert the multi-dimension to single-dimension
    weight.shape = (-1, 1)

    #drawHist(weight)

    for w in weight:
        pf.write('%.10f, ' % w)
        #pf.write('%s, ' % float(w))
        #weight_final.append(float(w))

    ## write biases
    #pf.write('\n\n' + param_name + '_bias:\n\n')
    ## convert the multi-dimension to single-dimension
    #bias.shape = (-1, 1)
    #for b in bias:
    #    pf.write('%.10f, ' % b)

    pf.write('\n\n')
    '''

#drawHist(weight_final)
#weight_final.sort()
#drawPlot(weight_final)
#"""

net.save('./python_modified.caffemodel')

pf.close
