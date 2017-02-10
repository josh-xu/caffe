make clean
make -j4
rm output_CONV.log
rm output_FC.log
./mnist_step_train.sh
