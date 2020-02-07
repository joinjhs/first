# ISPL_Freshman_practice_

<MNIST classification>

 
I modified some of the skeleton code, the usage is as below.

1) create a new train set and a validation set
python main.py --process=write_valid --imagedir=./mnist/train

2) create a new test set
python main.py --process=write_test --imagedir=./mnist/test

3) train
python main.py --process=train

4) test
python main.py --process=test


usage: 

    python main.py --process=write --imagedir=./mnist/train --datadir=./mnist/train_tfrecord

    or

    python main.py --process=train --datadir=./mnist/train_tfrecord --val_datadir=./mnist/val_tfrecord --epoch=1 --lr=1e-3 --ckptdir=./ckpt --batch=100 --restore=False
