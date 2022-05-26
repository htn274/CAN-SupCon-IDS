"""
Used to convert .csv into tfrecord format
"""
import os
import pandas as pd
import numpy as np
import glob
import swifter
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import argparse

attributes = ['Timestamp', 'canID', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'Flag']
def fill_flag(sample):
    if not isinstance(sample['Flag'], str):
        col = 'Data' + str(sample['DLC'])
        sample['Flag'], sample[col] = sample[col], sample['Flag']
    return sample

   
def serialize_example(x, y):
    """converts x, y to tf.train.Example and serialize"""
    #Need to pay attention to whether it needs to be converted to numpy() form
    id_seq, data_seq, data_histogram = x
    id_seq = tf.train.Int64List(value = np.array(id_seq).flatten())
    data_seq = tf.train.Int64List(value = np.array(data_seq).flatten())
    data_histogram = tf.train.Int64List(value = np.array(data_histogram).flatten())
    label = tf.train.Int64List(value = np.array([y]))
    features = tf.train.Features(
        feature = {
            "id_seq": tf.train.Feature(int64_list = id_seq),
            "data_seq": tf.train.Feature(int64_list = data_seq),
            "data_histogram": tf.train.Feature(int64_list = data_histogram),
            "label" : tf.train.Feature(int64_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        X = (row['id_seq'], row['data_seq'], row['data_histogram'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close()    
    
def preprocess(file_name, attack_id, window_size = 29, strided_size = 29):
    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = pd.read_csv(file_name, header=None, names=attributes)
    print("Reading {}: done".format(file_name))
    df = df.sort_values('Timestamp', ascending=True)
    df = df.swifter.apply(fill_flag, axis=1) # Paralellization is faster
    # Change data from hex string to int
    num_data_bytes = 8
    for x in range(num_data_bytes):
        df['Data'+str(x)] = df['Data'+str(x)].map(lambda x: int(x, 16), na_action='ignore')
    # Change can id from hex string to binary 29-bits length
    df['canID'] = df['canID'].apply(int, base=16).apply(bin).str[2:]\
                            .apply(lambda x: x.zfill(29)).apply(list)\
                            .apply(lambda x: list(map(int, x)))
    df = df.fillna(0)
    data_cols = ['Data{}'.format(x) for x in range(num_data_bytes)]
    df[data_cols] = df[data_cols].astype(int) 
    df['Data'] = df[data_cols].values.tolist()
    df['Flag'] = df['Flag'].apply(lambda x: True if x=='T' else False)
    print("Pre-processing: Done")
    
    as_strided = np.lib.stride_tricks.as_strided
    output_shape = ((len(df) - window_size) // strided_size + 1, window_size)
    canid = as_strided(df.canID, output_shape, (8*strided_size, 8))
    data = as_strided(df.Data, output_shape, (8*strided_size, 8)) #Stride is counted by bytes
    label = as_strided(df.Flag, output_shape, (1*strided_size, 1))

    df = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()), 
        'data_seq': pd.Series(data.tolist()),
        'label': pd.Series(label.tolist())
    }, index= range(len(canid)))
    df['data_histogram'] = df['data_seq'].apply(lambda x: np.histogram(np.array(x), bins=256)[0])
    df['label'] = df['label'].apply(lambda x: attack_id if any(x) else 0)
    print("Aggregating data: Done")
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])
    return df[['id_seq', 'data_seq', 'data_histogram', 'label']].reset_index().drop(['index'], axis=1)

def main(indir, outdir, car_model, attacks, window_size, strided):
    print(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_info = {}
    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}_dataset_{}.txt'.format(indir, attack, car_model)
        df = preprocess(finput, attack_id + 1, window_size, strided)
        print("Writing...................")
        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] != 0]
        df_normal = df[df['label'] == 0]
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
        
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="../Data/Car-Hacking")
    parser.add_argument('--outdir', type=str, default="../Data/TFRecord")
    parser.add_argument('--car_model', type=str)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--strided', type=int, default=None)
    parser.add_argument('--attack_type', type=str, default="all", nargs='+')
    args = parser.parse_args()
    
    if args.attack_type == 'all':
        attack_types = ['Flooding', 'Fuzzy', 'Malfunction']
        # attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
    else:
        attack_types = [args.attack_type]
    
    if args.strided == None:
        args.strided = args.window_size
        
    indir = os.path.join(args.indir, args.car_model)
    outdir = args.outdir + 'TFrecord_{}_w{}_s{}'.format(args.car_model, args.window_size, args.strided)
    main(indir, outdir, args.car_model, attack_types, args.window_size, args.strided)
        
    
    
