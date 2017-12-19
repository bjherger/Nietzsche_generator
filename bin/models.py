import keras
import numpy
from keras import Model, Sequential
from keras import backend as K
from keras.layers import Dense, Flatten, Embedding, LSTM, Activation, Reshape, Lambda
from keras.optimizers import RMSprop
from tensorflow import one_hot

import lib


def ff_model(embedding_input_dim, embedding_output_dim, X, y):
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
    x = Dense(128, activation='linear')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='linear')(x)
    x = output_layer(x)

    char_model = Model(sequence_input, x)
    char_model.compile(optimizer='Adam', loss='categorical_crossentropy')

    return char_model


def rnn_embedding_model(embedding_input_dim, embedding_output_dim, X, y):
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
    x = embedding_layer(sequence_input)
    # x = Flatten()(x)
    # x = Reshape()(x)
    x = LSTM(128)(x)
    x = output_layer(x)

    char_model = Model(sequence_input, x)
    char_model.compile(optimizer='Adam', loss='categorical_crossentropy')

    return char_model

def rnn_model(embedding_input_dim, embedding_output_dim, X, y):
    if len(X.shape) >= 2:
        input_length = int(X.shape[1])
    else:
        input_length = 1

    nb_classes = numpy.max(X)

    sequence_input = keras.Input(shape=(input_length,), dtype='int32', name='char_input')

    x_ohe = Lambda(K.one_hot,
                   arguments={'num_classes': nb_classes}, output_shape=(input_length,nb_classes))

    # Create output layer
    softmax_output_dim = len(y[0])
    output_layer = Dense(units=softmax_output_dim, activation='softmax')

    # Create model architecture
    x = x_ohe(sequence_input)
    # x = Flatten()(x)
    # x = Reshape()(x)
    x = LSTM(128)(x)
    x = output_layer(x)

    optimizer = RMSprop(lr=.01)
    char_model = Model(sequence_input, x)
    char_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return char_model