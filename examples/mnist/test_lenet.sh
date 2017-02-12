#!/usr/bin/env sh
set -e

#./build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_iter_10000.caffemodel $@
#./build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=../ristretto/examples/mnist/ristretto_lenet_iter_10000.caffemodel $@
../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_iter_20000.caffemodel -gpu=0 --iterations=200 $@
