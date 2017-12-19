import keras
from keras import Model, Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, Activation
from keras.optimizers import RMSprop

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


def rnn_model(embedding_input_dim, embedding_output_dim, X, y):
    maxlen = lib.get_conf('ngram_len')

    rnn_model = Sequential()
    rnn_model.add(LSTM(128, input_shape=(maxlen, embedding_input_dim)))
    rnn_model.add(Dense(embedding_input_dim))
    rnn_model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    rnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return rnn_model