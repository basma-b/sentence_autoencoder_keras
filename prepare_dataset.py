# -*- encoding:utf-8 -*-

"""
here we prepare the dataset in form of .pkl files
"""

from __future__ import print_function

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import re
np.random.seed(1337)
import sys


# transforms a csv file to pkl

def clean_text(text):
    text = text.lower()
    text = text.replace("\r","")
    text = text.replace("\n","")
    text = text.strip()
    return text

def load_data(filename):
    rows = pd.read_csv(filename, keep_default_na=False)
    sentences = []
    
    for i in range(rows.shape[0]):
        text = rows.loc[i].ReviewText
        if text != "":
            sentences.append(clean_text(text))
    return sentences

def main():
    
    embedding_file = sys.argv[1]
    param_file = sys.argv[2]
    input_file = sys.argv[3]

    # loading the data
    sentences = load_data(input_file)   
    print("Data loaded")
    
    #tokenizer = Tokenizer()
    tokenizer = Tokenizer(5000) # we keep the 5000 most frequent words of our vocabulary
    
    tokenizer.fit_on_texts(sentences)

    #MAX_SEQUENCE_LENGTH = max([len(seq) for seq in sentences])
    #print(MAX_SEQUENCE_LENGTH)
    MAX_SEQUENCE_LENGTH = 100 # we keep the last 100 words of every sentence

    #MAX_NB_WORDS = len(tokenizer.word_index) + 1
    MAX_NB_WORDS = 5000
    
    word_index = tokenizer.word_index
    
    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))
    
    sentences = tokenizer.texts_to_sequences(sentences)
    sentences = pad_sequences(sentences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # loading the embedding file
    
    print('Now indexing word vectors...')

    embeddings_index = {}
    f = open(embedding_file, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
        embeddings_index[word] = coefs
    f.close()
    
    embedding_size = len(embeddings_index.values()[0])
    
    maximum = 4.0
    
    print("Now constructing embedding matrix...")
    
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words , embedding_size))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None:
            # we normalize word embedding vectors 
            embedding_matrix[i] = [float(x)/maximum for x in embedding_vector] #embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(-0.25, 0.25, embedding_size)
            
    # we need to represent every input sequence using word embeddings
    temp = np.zeros((sentences.shape[0], MAX_SEQUENCE_LENGTH, embedding_size))
    for i in range(sentences.shape[0]):
        # for each sequence
        for j in range(MAX_SEQUENCE_LENGTH):
            # for each word of the sequence, get it embedding vector from the embedding matrix
            temp[i][j] = embedding_matrix[sentences[i][j]]
            
    
    sentences = temp
    print(sentences.shape)
    
    print("Now dumping")

    pickle.dump(sentences, open(input_file + ".pkl", "wb"))
    pickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open(param_file + ".pkl", "wb"))
    
if __name__ == '__main__':
    main()
