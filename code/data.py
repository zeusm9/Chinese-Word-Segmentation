import collections
from tensorflow.python.keras.utils import to_categorical
import numpy as np


def dictionary(data):
    # Unigram vocabulary, return a unigram to id vocabulary and the reversed vocabulary
    unigrams = []
    for line in data:
        for unigram in line:
            unigrams.append(unigram)
    count = collections.Counter(unigrams)

    word_to_id = dict()
    word_to_id["<UNK>"] = 1  # Tag <UNK> needed for OOV unigrams
    for unigram in count:
        word_to_id.update({unigram: len(word_to_id) + 1})
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


def bigrams_vocab(data):
    # Unigram vocabulary, return a bigram to id vocabulary and the reversed vocabulary
    bigrams = []
    for line in data:
        for bigram in line:
            bigrams.append(bigram)
    count = collections.Counter(bigrams)

    word_to_id = dict()
    word_to_id["<UNK>"] = 1  # Tag <UNK> needed for OOV bigrams
    word_to_id["<END>"] = 2  # Tag <END> needed to take the last bigram in each line
    for bigram in count:
        word_to_id.update({bigram: len(word_to_id) + 1})
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


def label_dict():
    # Dictionary of the labels
    label_to_id = dict()
    label_to_id["B"] = 0
    label_to_id["I"] = 1
    label_to_id["E"] = 2
    label_to_id["S"] = 3

    id_to_label = {v: k for k, v in label_to_id.items()}
    return label_to_id, id_to_label


def words_to_id(data, vocab):
    # Mapping of unigrams and bigrams to integers, returns a list of numpy arrays of integers
    output = []
    for line in data:
        paragraph = []
        for word in line:
            if word in vocab:
                paragraph.append(vocab[word])
            else:
                paragraph.append(vocab['<UNK>'])  # OOV words are mapped as unknown
        paragraph = np.array(paragraph)
        output.append(paragraph)
    return output


def labels_to_id(data, vocab):
    # Mapping of labels in BIES format to integers, return a list of binary matrices
    output = []
    for line in data:
        paragraph = []
        for word in line:
            paragraph.append(vocab[word])
        paragraph = np.array(paragraph)
        paragraph = paragraph.flatten()
        # transform the numpy array in binary matrix representation
        paragraph = to_categorical(paragraph, num_classes=4)
        output.append(paragraph)
    return output


def split_into_bigrams(data):
    # splits the data in bigrams, returns a list fo lists of bigrams
    bigrams = []
    for line in data:
        sentence = []
        for i in range(len(line) - 1):
            bigram = line[i] + line[i + 1]
            sentence.append(bigram)
        end = line[len(line) - 1] + "<END>"
        sentence.append(end)
        bigrams.append(sentence)
    return bigrams


def print_predictions(predictions, filename):
    # prints the predicted labels in BIES format in a file
    f = open(filename, 'w')
    for prediction in predictions:
        f.write(prediction)
        f.write("\n")
    f.close()
