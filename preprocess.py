import tensorflow as tf
import numpy as np
import glob
import random
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw

def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    feature = {'encoded' : _bytes_feature(img),
               'label' : _int64_feature(lab)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()



def write_tfrecord(imagedir, datadir, process): # Put a new input 'process', in order to make validation set easier.
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    if process == 'write_test' :

        writer = tf.python_io.TFRecordWriter(datadir+'/test.tfrecord')

        for i in range(10):
            filenames = glob.glob(imagedir + '/' + str(i) + '/*')
            for j in range(len(filenames)) :
                filename = filenames[j]
                #img_data = open(filename, 'rb').read()

                lab = i

                example = make_example(_image_as_bytes(filename), lab)
                writer.write(example)
        writer.close()

    elif process == 'write_valid' :

        writer_t = tf.python_io.TFRecordWriter(datadir + '/train.tfrecord')
        writer_v = tf.python_io.TFRecordWriter(datadir + '/valid.tfrecord')

        array_to_be_shuffled = []
        labels_to_be_shuffled = []

        for i in range(10):
            filenames = glob.glob(imagedir + '/' + str(i) + '/*')
            lab = i
            for j in range(len(filenames)):
                array_to_be_shuffled.append(filenames[j])
                labels_to_be_shuffled.append(lab)

        merged_array = np.rec.fromarrays([array_to_be_shuffled, labels_to_be_shuffled], names=('filename', 'label'))
        random.shuffle(merged_array)

        for i in range(len(merged_array)):

            #img_raw = _image_as_bytes(merged_array[i]['filename'])
            #label = merged_array[i]['label']
            img_raw, label = merged_array[i]
            #print(img_raw)
            example = make_example(_image_as_bytes(img_raw), label)
            prob = random.randrange(0, 100)  # to decide whether the image to be put into the validation set or the train set.
            if prob > 30 :
                writer_t.write(example)
            else :
                writer_v.write(example)

        writer_t.close()
        writer_v.close()





def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """
    filenames = glob.glob(folder)#+'/record.tfrecord')
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    key_to_feature = {'encoded' : tf.FixedLenFeature([], tf.string, default_value=''),
                      'label' : tf.FixedLenFeature([], tf.int64, default_value=0)}

    features = tf.parse_single_example(serialized_example, features=key_to_feature)


    #decode data
    img = tf.decode_raw(features['encoded'], tf.uint8)

    img = tf.cast(img, tf.float32)
    img = img / 255.0
    img = tf.reshape(img, [28, 28, 1])

    lab = tf.cast(features['label'], tf.int32)
    lab = tf.one_hot(lab, 10)


    #mini-batch examples queue
    #min_after_dequeue = 200
    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch, capacity=10*batch, num_threads=1, min_after_dequeue=batch*2)

    return img, lab


