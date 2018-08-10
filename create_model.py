# __author__=   'Sargam Modak'

import os
import json
import pickle
import logging
import numpy as np
from keras.models import Model
from keras.initializers import Constant
from keras import regularizers
from keras.layers import Embedding, Input, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, Dense, BatchNormalization, Dropout, LSTM


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_NUM_WORDS = 5000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)


def get_model():
    """
    returns the model
    :return:
    """
    
    embeddings_index = {}
    
    # load glove vector and create dictionary key:word value:corresponding 100d vector
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # load tokenizer
    tokenizer = pickle.load(file=open(name='util_files/tokenizer'))
    word_index = tokenizer.word_index
    
    cls2id = dict(json.load(fp=open(name='util_files/cls2id.json')))
    num_classes = len(cls2id)

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    
    embedded_sequences = embedding_layer(sequence_input)
    
    x = LSTM(200)(embedded_sequences)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    """
    x = Conv1D(128, 5, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(192, 5, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(3)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(256, 5, activation='tanh')(x)
    x = GlobalMaxPooling1D()(x)
    """
    
    x = Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(sequence_input, preds)
    print model.summary()
    
    return model
    
if __name__ == '__main__':
    get_model()
