#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import io

import keras
import numpy
import pandas
import sys
from keras import Input, Model
from keras.layers import Embedding, Flatten, Dense
from keras.utils import get_file
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

import lib


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    chars = extract()
    chars, encoded_chars, encoder, X = transform(chars)
    chars, encoded_chars, encoder, ohe, char_model = model(chars, encoded_chars, encoder, X)

    test_snippets = ['the ex', 'helpf', 'eak pl', 'ow: on', 'nvestig']

    for snippet in test_snippets:
        print "'", snippet, lib.model_predict(encoder, ohe, char_model, snippet), "'"

    load()
    pass


def extract():
    logging.info('Begin extract')

    # Check if Niezsche is available. If not, download.
    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

    # Pull text from file
    chars = io.open(path, encoding='utf-8').read().lower()

    # Convert to array of characters
    chars = list(chars)

    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return chars


def transform(chars):
    logging.info('Begin transform')
    # Filter characters
    chars = map(lambda x: x.lower(), chars)
    pre_filter_len = len(chars)
    filter(lambda x: x in lib.legal_characters(), chars)
    post_filter_len = len(chars)
    logging.info('Filtered with legal characters. Length before: {}. Length after: {}'.format(pre_filter_len,
                                                                                              post_filter_len))

    # Convert character ngrams into label representations
    encoder = LabelEncoder()
    encoded_chars = encoder.fit_transform(chars)

    # Create X, containing character ngrams
    ngrams = lib.find_ngrams(encoded_chars, lib.get_conf('ngram_len'))
    X = numpy.matrix(ngrams)

    lib.archive_dataset_schemas('transform', locals(), globals())

    return chars, encoded_chars, encoder, X



def model(chars, encoded_chars, encoder, X):
    logging.info('Begin model')



    # Create y, containing next character after window
    ohe = OneHotEncoder(sparse=False)
    y_intermediate = map(lambda x: [x], encoded_chars[lib.get_conf('ngram_len') - 1:])
    y = ohe.fit_transform(y_intermediate)

    logging.info('Created X with shape {}, and y with shape: {}'.format(X.shape, y.shape))

    # Create embedding
    embedding_input_dim = len(encoder.classes_)
    embedding_output_dim = min((embedding_input_dim +  1)/2, 50)
    if len(X.shape) >= 2:
        embedding_input_length = int(X.shape[1])
    else:
        embedding_input_length = 1

    sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32', name='char_input')

    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                input_length=embedding_input_length,
                                trainable=True,
                                name='char_embedding')

    # Create output layer
    softmax_output_dim = len(y[0])
    output_layer = Dense(units=softmax_output_dim, activation='softmax')

    # Create model architecture
    embedded_sequences = embedding_layer(sequence_input)
    x = Flatten()(embedded_sequences)
    x = output_layer(x)

    char_model = Model(sequence_input, x)
    char_model.compile(optimizer='Adam', loss='categorical_crossentropy')

    # Train model
    char_model.fit(X, y, batch_size=2048, validation_split=.2, epochs=3)

    lib.archive_dataset_schemas('model', locals(), globals())
    logging.info('End model')
    return chars, encoded_chars, encoder, ohe, char_model


def load():
    logging.info('Begin load')

    # TODO Serialize encoder

    # TODO Serialize model



    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
