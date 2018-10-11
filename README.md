# Sequence autoencoder in keras

This repository contains an implementation of text autoencoder in Keras.

# Dataset

I downloaded a toy dataset from https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews on which we will train the autoencoder. You can replace this dataset with anything else and put it in **data/** repository.

# Instructions
1. Data preprocessing:

'''
python prepare_dataset.py ~/udc_coherence/dual_encoder_udc/embeddings/glove.840B.300d.txt  data/params data/reviews.csv
'''

Training the model

python train_autoencoder.py --seq_length 100 --n_epochs 100 --optimizer adam --input_data data/reviews.csv.pkl --model_fname models/autoencoder.h5 --n_epochs 100

# Requirements

- Python 2.7
- Keras 2.0.8
- Theano 1.0.0
- Numpy 1.14.0

# Materials

- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Keras autoencoder example — sentence vectors](https://www.safaribooksonline.com/library/view/deep-learning-with/9781787128422/ee2fe540-56ff-4d05-b2f2-4e1e35b9d47f.xhtml)
- [Implementing Autoencoders in Keras: Tutorial](https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial)