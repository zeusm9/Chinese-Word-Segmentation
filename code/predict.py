import os
from argparse import ArgumentParser
from tensorflow.python.keras.models import load_model
from code import data as dt, preprocess as pre,model as mo

TRAIN_FILE = './resources/msr_training.utf8'
TEST_FILE = './resources/msr_test_gold.utf8'
INPUT_TRAINING = './resources/input_training.utf8'
OUTPUT_TRAINING = './resources/output_training.utf8'
INPUT_TEST = './resources/input.utf8'
OUTPUT_TEST = './resources/output.utf8'

def predict_line(line,bigram_line,model,vocab):
  #makes a prediction and takes the argmax, return the string of the predicted line
  prediction = model.predict([[line],[bigram_line]])
  prob = prediction.argmax(-1)[0]
  #takes the respective labels
  text = [vocab[p] for p in prob]
  predict_text = "".join(text)
  return predict_text

def make_predictions(model, testX, id_to_label,testX_bigram,output_path):
  predictions = []
  for i in range(len(testX)):
      predicted = predict_line(testX[i],testX_bigram[i],model,id_to_label)
      predictions.append(predicted)
  #print predictions on file
  dt.print_predictions(predictions, output_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    model = load_model(os.path.join(resources_path, 'my_model2.h5'))
    pre.generate_files(TRAIN_FILE, INPUT_TRAINING, OUTPUT_TRAINING)
    pre.generate_files(input_path,INPUT_TEST ,OUTPUT_TEST)

    testX = pre.read_data(INPUT_TEST)
    trainX = pre.read_data(INPUT_TRAINING)
    testX_bigram = dt.split_into_bigrams(testX)
    trainX_bigram = dt.split_into_bigrams(trainX)

    word_to_id, id_to_word = dt.dictionary(trainX)
    label_to_id, id_to_label = dt.label_dict()
    bigrams_to_id, id_to_bigrams = dt.bigrams_vocab(trainX_bigram)

    testX = dt.words_to_id(testX, word_to_id)
    testX_bigram = dt.words_to_id(testX_bigram, bigrams_to_id)


    make_predictions(model, testX, id_to_label, testX_bigram, output_path)
    pass


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
