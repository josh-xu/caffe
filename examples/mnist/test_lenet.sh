#!/usr/bin/env sh
set -e

CMD="../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_iter_${1}.caffemodel -gpu=0 --iterations=200";

#./build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_iter_10000.caffemodel $@
#./build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=../ristretto/examples/mnist/ristretto_lenet_iter_10000.caffemodel $@

#../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_1b_4mask_models/lenet_iter_20000.caffemodel -gpu=0 --iterations=200 $@

#../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_deep_compression_models/lenet_iter_20000.caffemodel -gpu=0 --iterations=200 $@
#../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_deep_compression_models/lenet_iter_28000.caffemodel -gpu=0 --iterations=200 $@
#../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_deep_compression_models/lenet_iter_30000.caffemodel -gpu=0 --iterations=200 $@

#../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_orig_models/lenet_iter_10000.caffemodel -gpu=0 --iterations=200 $@

#../caffe_github/build/tools/caffe test -model=examples/mnist/lenet_train_test.prototxt -weights=examples/mnist/lenet_iter_20000.caffemodel -gpu=0 --iterations=200 $@
eval $CMD;
