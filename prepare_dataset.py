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

def clean_text (text):
    text = text.lower()
    return text

def load_data(filename):
    train = pd.read_csv(csv_file)
    rows = train.shape[0]
    context = []
    response = []
    label = []
    for i in range(rows):
        context.append(clean_text(train.loc[i].Context))
        #print(i, train.loc[i].Response)
        response.append(clean_text(train.loc[i].Response))
        label.append(int(train.loc[i].Label))
    return context, response, label

def main():
    
    param_file = sys.argv[1]
    input_file_file = sys.argv[2]

    # loading train data
    train_c, train_r, train_l = build_data(train_file)   
    print("Finish train")

    
    tokenizer = Tokenizer()
    #tokenizer = Tokenizer(10000)
    
    tokenizer.fit_on_texts(train_c + train_r + dev_c + dev_r + test_c + test_r )

    #MAX_SEQUENCE_LENGTH = max([len(seq) for seq in train_c + train_r
                                                    #+ test_c + test_r
                                                    #+ dev_c + dev_r])
                                                    
    MAX_SEQUENCE_LENGTH = 100

    #MAX_NB_WORDS = len(tokenizer.word_index) + 1
    MAX_NB_WORDS = 10000
    
    word_index = tokenizer.word_index
    
    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))
    
    
    test_c = tokenizer.texts_to_sequences(test_c)
    
    test_r = pad_sequences(test_r, maxlen=MAX_SEQUENCE_LENGTH)

    print("Now dumping")

    pickle.dump([test_c, test_r], open(test_file + ".seq100.pkl", "wb"))
    pickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open(param_file + ".seq100.pkl", "wb"))
    
if __name__ == '__main__':
    main()
