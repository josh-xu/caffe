#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_15000.prototxt \
    --snapshot=examples/mnist/lenet_iter_10000.solverstate \
    $@

 ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_30000.prototxt \
    --snapshot=examples/mnist/lenet_iter_15000.solverstate \
    $@
if false; then
 ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_13000.prototxt \
    --snapshot=examples/mnist/lenet_iter_12000.solverstate \
    $@

     ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_14000.prototxt \
    --snapshot=examples/mnist/lenet_iter_13000.solverstate \
    $@

     ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_15000.prototxt \
    --snapshot=examples/mnist/lenet_iter_14000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_16000.prototxt \
    --snapshot=examples/mnist/lenet_iter_15000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_17000.prototxt \
    --snapshot=examples/mnist/lenet_iter_16000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_18000.prototxt \
    --snapshot=examples/mnist/lenet_iter_17000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_19000.prototxt \
    --snapshot=examples/mnist/lenet_iter_18000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_20000.prototxt \
    --snapshot=examples/mnist/lenet_iter_19000.solverstate \
    $@
fi

