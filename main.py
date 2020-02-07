import numpy as np
import glob
import os
import tensorflow as tf
from preprocess import *
from classifier import Classifier
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('process', 'write_valid', 'which process you want to do: write(write tfrecord file), train(train model), test(test model)')
tf.flags.DEFINE_string('imagedir', './mnist/train', 'directory where images are stored in')
tf.flags.DEFINE_string('datadir', './mnist/tfrecord', 'directory where tfrecord files are stored in')

########tfrecords for the validation set, train set, and test set will be stored in the same directory, 'tfrecord'.

#tf.flags.DEFINE_string('val_datadir', '../mnist/val_tfrecord', 'directory where validation tfrecord files are stored in')
tf.flags.DEFINE_string('ckptdir', './ckpt', 'checkpoint directory')

tf.flags.DEFINE_string('gpu', '0', 'gpu number to be used')
tf.flags.DEFINE_bool('restore', False, 'restore pre-trained model or not')

tf.flags.DEFINE_float('lr', 1e-3, 'initial learning rate')
tf.flags.DEFINE_integer('epoch', 3, 'total number of epochs for training')
tf.flags.DEFINE_integer('batch', 100, 'batch size')


def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if FLAGS.process == 'write_test' or FLAGS.process == 'write_valid':
        write_tfrecord(FLAGS.imagedir, FLAGS.datadir, FLAGS.process)

    else:
        classifier = Classifier(FLAGS)
        if FLAGS.process == 'train':
            classifier.train()
        elif FLAGS.process == 'test':
            classifier.test()


if __name__ == "__main__":
    tf.app.run()