# Sentence autoencoder with keras

Sentence autoencoder in Keras

# Dataset
I downloaded a toy dataset from https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews where we will train the autoencoder on the reviews. You can replace this dataset with anything else.

# Instruction

First:
    Preprocessing of data
        python prepare_dataset.py ~/udc_coherence/dual_encoder_udc/embeddings/glove.840B.300d.txt  data/params data/reviews.csv
Training the model

python train_autoencoder.py --seq_length 100 --n_epochs 100 --optimizer adam --input_data data/reviews.csv.pkl --model_fname models/autoencoder.h5 --n_epochs 100