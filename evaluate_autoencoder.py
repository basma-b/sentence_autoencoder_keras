# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle
import keras
from keras.utils import *
from keras.layers import *
from keras.models import  *
from keras.callbacks import ModelCheckpoint
from utilities import my_callbacks
import argparse
from utilities.data_helper import *
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import os
from keras import backend as K
from scipy.spatial.distance import cdist


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))

def main():
    
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--input_data', type=str, default='data/input.pkl', help='Input data')
    parser.add_argument('--output_data', type=str, default='data/output.pkl', help='Output data')
    parser.add_argument('--model_fname', type=str, default='models/autoencoder.h5', help='Model filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print ('Model args: ', args)
    
    np.random.seed(args.seed)
    
    print("Starting...")
    
    sequences = pickle.load(open(args.input_data, 'rb'))
    print(sequences.shape)
    
    print('Found %s sequences.' % len(sequences))
    
    print("Now loading the autoencoder...")
    autoencoder = load_model(args.model_fname)
    
    encoder = Model(autoencoder.input, autoencoder.get_layer("encoder").output)

    xtest = sequences
    ytest = autoencoder.predict(xtest)
    
    Xvec = encoder.predict(xtest)
    Yvec = encoder.predict(ytest)
    
    cosims = np.zeros((Xvec.shape[0]))
    
    # Compute the cosine similarity between the 
    for rid in range(Xvec.shape[0]):
        cosims[rid] = cosine_similarity(Xvec[rid], Yvec[rid])
    
    print("The average cosine similarity is ", np.mean(cosims))
    
    pickle.dump(Xvec, open(args.output_data, "wb"))
    
if __name__ == "__main__":
    main()