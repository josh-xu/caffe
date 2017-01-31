#!/bin/bash

./build/examples/cpp_classification/classification.bin ./examples/mnist/lenet.prototxt ./examples/mnist/lenet_iter_20000.caffemodel ./data/mnist/mnist_mean.binaryproto ./data/mnist/mnist_words.txt ./data/mnist/test/test_${1}.bmp

open ./data/mnist/test/test_${1}.bmp
