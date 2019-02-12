#!pip install -U tensorflow
#!pip install -U keras

import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import h5py
import os
import pandas as pd
import time
from pandas import DataFrame as df


#from IPython.display import HTML
from keras.models import model_from_json

import sys
import types
import pandas as pd


#Loading the evaluation data and labels to numpy objects
def load_data(hdf5_path):
    """
    Loads the data into numpy objects. 
    Input : Path to data.
    Output : Train/Test examples, corresponding labels, corresponding youtube video_id.
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        video_id_list = hf.get('video_id_list')
        x = np.array(x)
        y = list(y)
        video_id_list = list(video_id_list)
        
    return x, y, video_id_list

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.
    
def bool_to_float32(y):
    return np.float32(y)

(x, y, video_id_list) = load_data('/home/stride/audioset_predict/packed_features/eval.h5')
x = uint8_to_float32(x)		# shape: (N, 10, 128)
print(x.shape)
y = bool_to_float32(y)		# shape: (N, 527)

print(video_id_list[350])

#Building the model

import keras
from keras.models import Model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate)
import keras.backend as K
from keras.optimizers import Adam

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def max_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):

    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return (sample_num, freq_bins)

model_type = 'decision_level_multi_attention'
time_steps = 10
freq_bins = 128
classes_num = 527

# Hyper parameters
hidden_units = 1024
drop_rate = 0.5
batch_size = 500

# Embedded layers
input_layer = Input(shape=(time_steps, freq_bins))

a1 = Dense(hidden_units)(input_layer)
a1 = BatchNormalization()(a1)
a1 = Activation('relu')(a1)
a1 = Dropout(drop_rate)(a1)

a2 = Dense(hidden_units)(a1)
a2 = BatchNormalization()(a2)
a2 = Activation('relu')(a2)
a2 = Dropout(drop_rate)(a2)

a3 = Dense(hidden_units)(a2)
a3 = BatchNormalization()(a3)
a3 = Activation('relu')(a3)
a3 = Dropout(drop_rate)(a3)

# Pooling layers
if model_type == 'decision_level_max_pooling':

    cla = Dense(classes_num, activation='sigmoid')(a3)
    output_layer = Lambda(max_pooling, output_shape=pooling_shape)([cla])

elif model_type == 'decision_level_average_pooling':
   
    cla = Dense(classes_num, activation='sigmoid')(a3)
    output_layer = Lambda(
        average_pooling,
        output_shape=pooling_shape)(
        [cla])

elif model_type == 'decision_level_single_attention':

  
    cla = Dense(classes_num, activation='sigmoid')(a3)
    att = Dense(classes_num, activation='softmax')(a3)
    output_layer = Lambda(
        attention_pooling, output_shape=pooling_shape)([cla, att])

elif model_type == 'decision_level_multi_attention':
    
    cla1 = Dense(classes_num, activation='sigmoid')(a2)
    att1 = Dense(classes_num, activation='softmax')(a2)
    out1 = Lambda(
        attention_pooling, output_shape=pooling_shape)([cla1, att1])

    cla2 = Dense(classes_num, activation='sigmoid')(a3)
    att2 = Dense(classes_num, activation='softmax')(a3)
    out2 = Lambda(
        attention_pooling, output_shape=pooling_shape)([cla2, att2])

    b1 = Concatenate(axis=-1)([out1, out2])
    b1 = Dense(classes_num)(b1)
    output_layer = Activation('sigmoid')(b1)

elif model_type == 'feature_level_attention':

    cla = Dense(hidden_units, activation='linear')(a3)
    att = Dense(hidden_units, activation='sigmoid')(a3)
    b1 = Lambda(
        attention_pooling, output_shape=pooling_shape)([cla, att])

    b1 = BatchNormalization()(b1)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(drop_rate)(b1)

    output_layer = Dense(classes_num, activation='sigmoid')(b1)

else:
    raise Exception("Incorrect model_type!")

# Build model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

#Loading weights
model.load_weights('/home/stride/audioset_predict/models/main/balance_type=balance_in_batch/model_type=decision_level_multi_attention/final_weights.h5')

df_data_1 = pd.read_csv('/home/stride/audioset_predict/metadata/class_labels_indices.csv')
df_data_1.head()


df_data_2 = pd.read_csv('/home/stride/audioset_predict/metadata/eval_segments.csv')
df_data_2.head()

def dup_rows(a, index, num_dups=1):
    return np.insert(a, [index+1]*num_dups,a[index], axis=0)

def demo(v):
    """
    A function to demonstrate the audio classification.
    Input : a random number to retrieve a query video.
    Output : video embed string for the YouTube video.
    Prints : Top 10 class probabilities from the classifier.
    """
    test = np.load('App15.npy')
    test = test[0:2]
    test = dup_rows(test, 0, 4)
    test = dup_rows(test, 1, 4)
    test = test.reshape(1,10,128)
    test_data = x[v:v+1,10:]
    current_infer = model.predict(test)
    current_video = str(video_id_list[v],'utf-8')
    start_time = int(df_data_2.loc[df_data_2['video_id'] == current_video]['start_time'])
    video_string = '<iframe width="560" height="315" src="https://www.youtube.com/embed/'+current_video+'?autoplay=1&start='+str(start_time)+';autoplay=1 frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>'
    #print('Predicted Labels :\n')
    #print(df_data_1.loc[current_infer[0].argsort()[-10:][::-1]]['display_name'])
    #print('Ground Truth Labels :\n')
    #print(df_data_1.loc[y[v].argsort()[-5:][::-1]]['display_name'])
    #print(video_string,start_time)
    return video_string, test_data[0]

t0 = time.time()
#Try out a random number between 0 and 20000 
for i in range(100,101):
    video_number = i;
    t0 = time.time()	
    vid_string, test = demo(video_number)
    t1 = time.time()

print (t1-t0)



