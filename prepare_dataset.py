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
    rows = pd.read_csv(csv_file)
    sentences = []
    
    for i in range(rows.shape[0]):
        sentences.append(clean_text(rows.loc[i].ReviewText))
    
    return sentences

def main():
    
    param_file = sys.argv[1]
    input_file = sys.argv[2]

    # loading the data
    sentences = load_data(input_file)   
    print("Data loaded")
    
    tokenizer = Tokenizer()
    #tokenizer = Tokenizer(10000)
    
    tokenizer.fit_on_texts(sentences)

    #MAX_SEQUENCE_LENGTH = max([len(seq) for seq in sentences)
    MAX_SEQUENCE_LENGTH = 100

    #MAX_NB_WORDS = len(tokenizer.word_index) + 1
    MAX_NB_WORDS = 5000
    
    word_index = tokenizer.word_index
    
    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))
    
    sentences = tokenizer.texts_to_sequences(sentences)
    sentences = pad_sequences(sentences, maxlen=MAX_SEQUENCE_LENGTH)

    print("Now dumping")

    pickle.dump(sentences, open(input_file + ".pkl", "wb"))
    pickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open(param_file + ".pkl", "wb"))
    
if __name__ == '__main__':
    main()
