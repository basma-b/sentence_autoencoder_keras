# Sequence autoencoder in keras

This repository contains an implementation of text autoencoder in Keras.

## Dataset

I downloaded a toy dataset from https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews on which we will train the autoencoder. You can replace this dataset with anything else and put it in **data/** repository.

## Instructions
1. Make sure you put your embedding file in **embeddings** directory. In this example we use Glove vectors of size 300.

2. Data preprocessing: whatever is the format of your input, you should preprocess it. The script **prepare_dataset.py** uses Keras preprocessings to tokenize and pad input sequences and finally embedd all the sequences. Use the following instruction :

```
python prepare_dataset.py ~/udc_coherence/dual_encoder_udc/embeddings/glove.840B.300d.txt  data/params data/reviews.csv
```
3. Training: the **train_autoencoder.py** script supposes that your input data is already preprocessed embedded (ref. step 2). It will produce a model that we store at **models** directory. 

```
python train_autoencoder.py --seq_length 100 --n_epochs 100 --optimizer adam --input_data data/reviews.csv.pkl --model_fname models/autoencoder.h5 
```
4. Evaluation: run **evaluate_autoencoder.py** script to measure the autoencoder capacity to produce an output that is similar to the input sequences.

```
python evaluate_autoencoder.py --input_data data/reviews.csv.pkl --model_fname models/autoencoder.h5 --output_data data/encoded_reviews.pkl
```

## Requirements

- Python 2.7
- Keras 2.0.8
- Theano 1.0.0
- Numpy 1.14.0

## Materials

- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Keras autoencoder example — sentence vectors](https://www.safaribooksonline.com/library/view/deep-learning-with/9781787128422/ee2fe540-56ff-4d05-b2f2-4e1e35b9d47f.xhtml)
- [Implementing Autoencoders in Keras: Tutorial](https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial)