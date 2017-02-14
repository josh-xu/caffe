#!/usr/bin/env sh
set -e

make clean_model

./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_11000.prototxt \
    --snapshot=examples/mnist/lenet_iter_10000.solverstate \
    $@

 ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_12000.prototxt \
    --snapshot=examples/mnist/lenet_iter_11000.solverstate \
    $@

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

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_21000.prototxt \
    --snapshot=examples/mnist/lenet_iter_20000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_22000.prototxt \
    --snapshot=examples/mnist/lenet_iter_21000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_23000.prototxt \
    --snapshot=examples/mnist/lenet_iter_22000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_24000.prototxt \
    --snapshot=examples/mnist/lenet_iter_23000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_25000.prototxt \
    --snapshot=examples/mnist/lenet_iter_24000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_26000.prototxt \
    --snapshot=examples/mnist/lenet_iter_25000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_27000.prototxt \
    --snapshot=examples/mnist/lenet_iter_26000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_28000.prototxt \
    --snapshot=examples/mnist/lenet_iter_27000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_29000.prototxt \
    --snapshot=examples/mnist/lenet_iter_28000.solverstate \
    $@

    ./build/tools/caffe train \
    --solver=examples/mnist/lenet_solver_30000.prototxt \
    --snapshot=examples/mnist/lenet_iter_29000.solverstate \
    $@
