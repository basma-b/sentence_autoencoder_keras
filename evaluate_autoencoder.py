# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle
import keras
from keras.utils import *
from keras.layers import *
from keras.models *
from keras.callbacks import ModelCheckpoint
from utilities import my_callbacks
import argparse
from utilities.data_helper import *
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import os
from keras import backend as K
from scipy.spatial.distance import cdist


def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def compute_cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))


def clean_text(text):
    text = text.lower()
    return text

def main():
    
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--input_data', type=str, default='data/input.pkl', help='Input data')
    parser.add_argument('--input_params', type=str, default='data/params.pkl', help='Input paramaters')
    parser.add_argument('--model_fname', type=str, default='models/autoencoder.h5', help='Model filename')
    parser.add_argument('--embedding_file', type=str, default='embeddings/glove.840B.300d.txt', help='Embedding filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print ('Model args: ', args)
    
    np.random.seed(args.seed)
    
    print("Starting...")
    
    # loading the embedding file
    
    print('Now indexing word vectors...')

    embeddings_index = {}
    f = open(args.embedding_file, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
        embeddings_index[word] = coefs
    f.close()
    
    # loading the input parameters
    
    MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index = pickle.load(open(args.input_params, 'rb'))
    
    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))
    
    maximum = 4.0
    
    print("Now constructing embedding matrix...")
    
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words , args.emb_dim))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None:
            # we normalize word embedding vectors 
            embedding_matrix[i] = [float(x)/maximum for x in embedding_vector] #embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(-0.25, 0.25, args.emb_dim)
    
    print("Now loading data...")
    
    sequences = pickle.load(open(args.input_data, 'rb'))
    print(sequences.shape)
    
    # we need to represent every input sequence using word embeddings
    temp = np.zeros((sequences.shape[0], MAX_SEQUENCE_LENGTH, args.emb_dim))
    for i in range(sequences.shape[0]):
        # for each sequence
        for j in range(MAX_SEQUENCE_LENGTH):
            # for each word of the sequence, get it embedding vector from the embedding matrix
            temp[i][j] = embedding_matrix[sequences[i][j]]
    
    sequences = temp
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
        cosims[rid] = compute_cosine_similarity(Xvec[rid], Yvec[rid])
    
    print(np.mean(cosims))
    
    pickle.dump(Xvec, open(args.input_dir + "encoded_responses.pkl", "wb"))
    
if __name__ == "__main__":
    main()