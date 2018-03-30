"""
This library aims at preprocess the images.
The idea is to create TFRecords files with three features:
    - image : JPEG
    - people_id : int
    - people_name : string
"""

from random import shuffle
import glob
from scipy import misc
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import sys
"""
train_addrs = 'C:/Users/G551JW/Desktop/celebrity_faces/input_training/people/*/*'
# read addresses and labels from the 'train' folder
addrs = glob.glob(train_addrs)

labels = [addr.split('/')[-1].split('\\')[1] for addr in addrs]
#print(label)

def load_image(addr):
    img = misc.imread(addr, mode='RGB')
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list= tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = 'train.tfrecords'

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(addrs)):
    # print how many images are saved every 100 images
    if not i % 100:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()

    # Load the image
    img = load_image(addrs[i])

    label = int(labels[i])

    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()
"""
data_path = 'train.tfrecords'

with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([],tf.string ),
                'train/label': tf.FixedLenFeature([], tf.int64)
    }

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)

    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(5):
        img, lbl = sess.run([images, labels])

        img = img.astype(np.uint8)

        for j in range(6):
            plt.subplot(2, 3, j+1)
            plt.imshow(img[j, ...])
        plt.show()

    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()
