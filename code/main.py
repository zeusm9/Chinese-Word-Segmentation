import tensorflow.keras as K
from code import model as mo, data as dt, preprocess as pre
from matplotlib import pyplot
from tensorflow.python.keras.callbacks import EarlyStopping

TRAIN_FILE = './resources/msr_training.utf8'
TEST_FILE = './resources/msr_test_gold.utf8'
INPUT_TRAINING = './resources/input_training.utf8'
OUTPUT_TRAINING = './resources/output_training.utf8'
INPUT_TEST = './resources/input_test.utf8'
OUTPUT_TEST = './resources/output_test.utf8'

def main():
    # Hyperparameters
    BATCH_SIZE = 128
    EMBEDDING_SIZE = 128
    BIGRAMS_EMBEDDING_SIZE = 64
    HIDDEN_SIZE = 256
    EPOCHS = 20
    MAX_LENGTH = 200

    # generates the files needed for training input_training:chinese text without spaces , output_training:labels
    pre.generate_files(TRAIN_FILE, INPUT_TRAINING, OUTPUT_TRAINING)
    # files for test input_test :chinese text without spaces,output_test:labels
    pre.generate_files(TEST_FILE,INPUT_TEST , OUTPUT_TEST)

    # gets the data from files
    trainX = pre.read_data(INPUT_TRAINING)
    trainY = pre.read_data(OUTPUT_TRAINING)
    testX = pre.read_data(INPUT_TEST)
    testY = pre.read_data(OUTPUT_TEST)

    # takes bigrams
    trainX_bigram = dt.split_into_bigrams(trainX)
    testX_bigram = dt.split_into_bigrams(testX)

    # builds vocabularies
    word_to_id, id_to_word = dt.dictionary(trainX)
    label_to_id, id_to_label = dt.label_dict()
    bigrams_to_id, id_to_bigrams = dt.bigrams_vocab(trainX_bigram)

    VOCAB_SIZE = len(word_to_id)
    BIGRAMS_VOCAB_SIZE = len(bigrams_to_id)

    # mapping of words to integers
    trainX = dt.words_to_id(trainX, word_to_id)
    trainY = dt.labels_to_id(trainY, label_to_id)
    testX = dt.words_to_id(testX, word_to_id)
    testY = dt.labels_to_id(testY, label_to_id)
    trainX_bigram = dt.words_to_id(trainX_bigram, bigrams_to_id)
    testX_bigram = dt.words_to_id(testX_bigram, bigrams_to_id)

    # builds the batches generators
    train_generator = mo.generator(trainX, trainY, trainX_bigram, BATCH_SIZE, MAX_LENGTH)
    test_generator = mo.generator(testX, testY, testX_bigram, BATCH_SIZE, MAX_LENGTH)

    # builds the embedding matrices
    embedding_mtx_bigram = mo.embedding_matrix(bigrams_to_id, BIGRAMS_EMBEDDING_SIZE)
    embedding_mtx_unigram = mo.embedding_matrix(word_to_id, EMBEDDING_SIZE)

    # create and compiles the model
    model = mo.create_keras_model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, BIGRAMS_VOCAB_SIZE, BIGRAMS_EMBEDDING_SIZE,
                               embedding_mtx_unigram, embedding_mtx_bigram)

    train_steps = len(trainX) // BATCH_SIZE
    test_steps = len(testX) // BATCH_SIZE

    # early stopping to prevent overfitting
    ea = EarlyStopping(monitor='val_loss')
    # begins the train
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps,
                                  validation_data=test_generator,
                                  validation_steps=test_steps,
                                  epochs=EPOCHS,
                                  callbacks=[ea])
    # saves the model and the weights
    model.save_weights('./resources/my_model_weights2.h5')
    # plot loss and val loss functions
    pyplot.plot(history.history['loss'], 'r--')
    pyplot.plot(history.history['val_loss'], 'b-')
    pyplot.title('Model train vs validation loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epochs')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    # plot accuracy and val accuracy
    pyplot.plot(history.history['acc'], 'r--')
    pyplot.plot(history.history['val_acc'], 'b-')
    pyplot.title('Model train vs validation accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epochs')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()


if __name__ == '__main__':
    main()