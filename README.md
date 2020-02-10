# ISPL_Freshman_practice_

<MNIST classification>

 
I modified some of the skeleton code, the usage is as below.

0) unzip mnist.zip and create a new folder "tfrecord" to store tfrecord files.


1) create a new train set and a validation set

python main.py --process=write_valid --imagedir=./mnist/train

2) create a new test set

python main.py --process=write_test --imagedir=./mnist/test

3) train

python main.py --process=train

4) test

python main.py --process=test



