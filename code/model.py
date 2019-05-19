import tensorflow as tf
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input,Embedding,Bidirectional,LSTM,concatenate,Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

EMBEDDING_FILE = './resources/merge_sgns_bigram_char300.txt'

def embedding_matrix(word_index, EMBEDDING_DIM):
    '''prepares the embedding_matrix for pretrained chinese word embedding,
    each line contains a word and its vector.
    Each value is separated by space.'''
    embeddings_index = {}
    f = open(EMBEDDING_FILE,encoding='utf8')
    i = 0
    for line in f:
        # the first line of the file are metadata, not needed, we can skip
        if i == 0:
            i = i + 1
        else:
            values = line.split()
            word = values[0]  # takes the word
            coefs = np.array(values[1:EMBEDDING_DIM + 1])  # takes the first #EMBEDDING_DIM entries of the vector
            embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    # build the matrix according to my vocabulary
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def generator(X, y, bigram, batch_size, max_len):
    # the batch generator generates batches of 'batch_size' lines
    while True:
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            X_bigram_batch = bigram[start:end]
            # MAX_LENGTH is the maximum size of a line in the batch
            MAX_LENGTH = len(max(X_batch, key=len))
            # if MAX_LENGTH exceeds a constant length, cut to this length
            if MAX_LENGTH > max_len:
                MAX_LENGTH = max_len
            # pad sequences to MAX_LENGTH with zeros
            X_batch = pad_sequences(X_batch, truncating='post', padding='post',
                                                             maxlen=MAX_LENGTH,value=0)
            y_batch = pad_sequences(y_batch, truncating='post', padding='post', maxlen=MAX_LENGTH,value=0)
            X_bigram = pad_sequences(X_bigram_batch, truncating='post', padding='post',
                                                     maxlen=MAX_LENGTH,value=0)

            yield [X_batch, X_bigram], y_batch


def create_keras_model(vocab_size, embedding_size, hidden_size, bigrams_vocab_size, bigrams_embedding_size,
                       embedding_matrix_uni, embedding_matrix_bi):
    input1 = Input(shape=(None,))
    # embedding layer for unigrams with masking and pretrained weights
    embedding1 = Embedding(vocab_size + 1, embedding_size, weights=[embedding_matrix_uni], trainable=False,
                                    mask_zero=True)(input1)

    input2 = Input(shape=(None,))
    # embedding layer for bigrams with masking and pretrained weights
    embedding2 = Embedding(bigrams_vocab_size + 1, bigrams_embedding_size, weights=[embedding_matrix_bi],
                                    trainable=False, mask_zero=True)(input2)

    # concatenation of both the embedding layers
    concatenated = concatenate([embedding1, embedding2])
    # bidirectional wrapper for lstm
    bi1 = Bidirectional(LSTM(hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(
        concatenated)

    dense = Dense(4, activation="softmax")(bi1)
    model = Model(inputs=[input1, input2], outputs=dense)
    optimizer = Adam(0.002)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    print("Model built!")
    return model
