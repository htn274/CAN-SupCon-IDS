import tensorflow as tf
import numpy as np
import json
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import os
print('Tensorflow version', tf.__version__)

class TFwriter:
    def __init__(self, outdir, start_idx = 0):
        print('Writing to: ', outdir)
        self._outdir = outdir
        self._start_idx = start_idx
        
    def serialize_example(self, x, y):
        """converts x, y to tf.train.Example and serialize"""
        #Need to pay attention to whether it needs to be converted to numpy() form
        id_seq, data_seq = x
        id_seq = tf.train.Int64List(value = np.array(id_seq).flatten())
        data_seq = tf.train.Int64List(value = np.array(data_seq).flatten())
        #data_histogram = tf.train.Int64List(value = np.array(data_histogram).flatten())
        label = tf.train.Int64List(value = np.array([y]))
        features = tf.train.Features(
            feature = {
                "id_seq": tf.train.Feature(int64_list = id_seq),
                "data_seq": tf.train.Feature(int64_list = data_seq),
                #"data_histogram": tf.train.Feature(int64_list = data_histogram),
                "label" : tf.train.Feature(int64_list = label)
            }
        )
        example = tf.train.Example(features = features)
        return example.SerializeToString()

    def write(self, data, label):
        filename = os.path.join(self._outdir, str(self._start_idx)+'.tfrec')
        with tf.io.TFRecordWriter(filename) as outfile:
            outfile.write(self.serialize_example(data, label))
        self._start_idx += 1
        
def read_tfrecord(example, window_size):
    # window_size = 20
    data_bytes = 256
    feature_description = {
    'id_seq': tf.io.FixedLenFeature([window_size*29], tf.int64),
    'data_seq': tf.io.FixedLenFeature([window_size*8], tf.int64),
    'data_histogram': tf.io.FixedLenFeature([data_bytes], tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)


def write_tfrecord(dataset, tfwriter):
    for batch_data in iter(dataset):
        features = zip(batch_data['id_seq'], batch_data['data_seq'])
        for x, y in zip(features, batch_data['label']):
            tfwriter.write(x, y)
            
if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--strided', type=int) 
    parser.add_argument('--rid', type=int, default=1) 
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    print(args, unknown)
    # data dir patern: '../Data/TFRecord_ID_DATA_HIST_w{window_size}_s{strided_size}'
    if args.strided == None:
        args.strided = args.window_size
        
    # data_dir = '../Data/TFRecord_ID_DATA_HIST_w{}_s{}'.format(args.window_size, args.strided)
    data_dir = '../Data/TFRecord_w{}_s{}'.format(args.window_size, args.strided)
    out_dir = data_dir + '/{}'.format(args.rid)
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    data_info = json.load(open(data_dir + '/datainfo.txt'))
    train_writer = TFwriter(train_dir)
    val_writer = TFwriter(val_dir)
    
    train_ratio = 0.7
    batch_size = 1000

    total_train_size = 0
    total_val_size = 0

    for filename, dataset_size in data_info.items():
        print('Read from {}: {} records'.format(filename, dataset_size))
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(lambda x: read_tfrecord(x, args.window_size), 
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(50000)

        train_size = int(dataset_size * train_ratio)
        val_size = (dataset_size - train_size)
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
            
        
        # inputs = ([train_dataset, train_writer], [val_dataset, val_writer])
        # p = Pool(2)
        # p.map(write_tfrecord, inputs)
        write_tfrecord(train_dataset, train_writer)
        write_tfrecord(val_dataset, val_writer)
        
        total_train_size += train_size
        total_val_size += val_size
        
    print('Total training: ', total_train_size)
    print('Total validation: ', total_val_size)