#!/usr/bin/env sh
set -e

rm -rf examples/mnist/lenet_iter_20000.caffemodel

rm -rf examples/mnist/lenet_iter_20000.solverstate

./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver.prototxt \
    --snapshot=examples/mnist/lenet_iter_10000.solverstate \
    $@
