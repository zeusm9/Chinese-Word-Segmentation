# Chinese Word Segmentation
In this project it has been used the paper 'State-of-the-art ChineseWord Segmentation with Bi-LSTMs' to
build the model.
The built model consists in a stacking bidirectional LSTM that takes in input two concatenated embedding
layers, one for unigrams, the second one for bigrams. The output is a 4-way Dense layer that outputs the
probabilities of the BIES tags (See Figure 1).
The model has been tested on Google Colab on GPU.
