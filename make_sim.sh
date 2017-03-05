make clean
make -j6
./build/tools/caffe train --solver=examples/mnist/lenet_solver_29000.prototxt --snapshot=examples/mnist/lenet_iter_28000.solverstate $@
