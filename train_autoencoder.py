# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle
import keras
from keras.utils import *
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
import argparse
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import os

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--seq_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--input_data', type=str, default='data/input.pkl', help='Input data')
    parser.add_argument('--model_fname', type=str, default='models/autoencoder.h5', help='Model filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print ('Model args: ', args)
    
    np.random.seed(args.seed)
    
    print("Starting...")
    
    
    print("Now building the autoencoder...")
    
    # the inputs should be already embedded
    embedded_inputs = Input(shape=(args.seq_length, args.emb_dim))
    encoded_inputs = LSTM(args.hidden_size, name="encoder")(embedded_inputs)
    
    decoded = RepeatVector(args.seq_length)(encoded_inputs)
    decoded = LSTM(args.emb_dim, return_sequences=True)(decoded) 
    
    autoencoder = Model(embedded_inputs, decoded)
    autoencoder.compile(loss='mse', optimizer=args.optimizer) # we use 'mse' loss function to train our autoencoder
    
    print(autoencoder.summary())
    
    print("Now loading data...")
    
    sequences = pickle.load(open(args.input_data, 'rb'))
    print(sequences.shape)
    
    print('Found %s sequences.' % len(sequences))
    
    print("Now training the model...")
    
    # we save the autoencoder model during the training
    checkpoint = ModelCheckpoint(filepath=args.model_fname, save_best_only=True)
                
    autoencoder.fit(sequences, sequences,
            batch_size=args.batch_size, epochs=args.n_epochs, verbose=1, validation_split=0.2, callbacks=[checkpoint])
            
    
if __name__ == "__main__":
    main()