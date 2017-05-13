import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler
import codecs
import nltk
import numpy as np
from sys import argv

import xml.etree.ElementTree as elementTree

INSTANCE = ".//instance"

SENSEID = 'senseid'

PRODUCT = "product"
PRODUCT_LBL = 4

PHONE = "phone"
PHONE_LBL = 3

FORMATION = "formation"
FORMATION_LBL = 2

DIVISION = "division"
DIVISION_LBL = 1

CORD = "cord"
CORD_LBL = 0

number_of_args = 4


def get_full_data(data, labels):
    """
    Return concatenations of the data (all DB)
    second list contains all the labels
    :param data: list of list of data to concat
    :param labels: list of all labels according to given data
    :return: 
    """
    full_data = []
    full_labels = []

    for i, data_set in enumerate(data):
        for vec in data_set:
            full_data.append(vec)
            full_labels.append(labels[i])

    return full_labels, full_data


def get_data(data):
    """
    Get the categories data
    :param data: xml tree stracture
    :return: cord, division, formation, phone, product arrays
    """
    cord = []
    division = []
    formation = []
    phone = []
    product = []

    for inst in data.findall(INSTANCE):

        att = inst[0].attrib[SENSEID]

        for sentence in inst[1].getchildren():

            tokens = nltk.word_tokenize(sentence.text)
            tokens = ' '.join(tokens).replace("<s>","").replace("<\s>","").lower()
            if att == CORD:
                cord.append(tokens)
            elif att == DIVISION:
                division.append(tokens)
            elif att == FORMATION:
                formation.append(tokens)
            elif att == PHONE:
                phone.append(tokens)
            elif att == PRODUCT:
                product.append(tokens)

    return cord, division, formation, phone, product


def print_scores(scores):
    for score in scores:
        print(str(score[0]) + ": " + str(round(score[1], 2)) + '\n')


def first_question(test_tree, train_tree):
    """
    First question: implement logistic regression to recognize "line" category
    :param test_tree: test data , xml structure  
    :param train_tree: train data, xml structure
    :return: 
    """

    # Get train data by classification
    cord, division, formation, phone, product = get_data(train_tree)

    # Build train data and labels arrays
    full_labels, full_data = get_full_data([cord, division, formation, phone, product],
                                           [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])

    # Create bag of words using TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    X = vectorizer.fit_transform(full_data, full_labels)

    # Run logistic regression
    model = LogisticRegression()
    model.fit(X, full_labels)

    # Get test data by classification
    cord, division, formation, phone, product = get_data(test_tree)

    # Build test data and labels arrays
    full_test_labels, full_test_data = get_full_data([cord, division, formation, phone, product],
                                                     [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])
    # Trasnform test data to same shape of train data
    X_test = vectorizer.transform(full_test_data, full_test_labels)

    # Predict using logistic regression
    y_predict = model.predict(X_test)

    # Flatten full_test_labels, this are the y (true) labels
    y = np.ravel(full_test_labels)

    # Get f1_score and accuracy
    f_score = f1_score(y, y_predict, average='macro')
    accuracy = accuracy_score(y, y_predict)
    return accuracy, f_score


def second_question(test_tree, train_tree):
    """
    Second question: 
    :param test_tree: test data , xml structure  
    :param train_tree: train data, xml structure
    :return: 
    """

    # Get train data by classification
    cord, division, formation, phone, product = get_data(train_tree)

    # Build train data and labels arrays
    full_labels, full_data = get_full_data([cord, division, formation, phone, product],
                                           [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])

    # Load word2vec embeddings file
    result = KeyedVectors.load_word2vec_format("wiki.en.100k.vec", binary=False)

    # Get train data features by word2vec
    X_train = get_features(full_data, result)

    # Normalize features
    scaler = MaxAbsScaler()
    X_train_maxabs = scaler.fit_transform(X_train)

    # Run logistic regression
    model = LogisticRegression()
    model.fit(X_train_maxabs, full_labels)

    # Get test data by classification
    cord, division, formation, phone, product = get_data(test_tree)

    # Build test data and labels arrays
    full_test_labels, full_test_data = get_full_data([cord, division, formation, phone, product],
                                                     [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])

    # Get test data features by word2vec
    X_test = get_features(full_test_data, result)

    # Normalize features
    X_test_maxabs = scaler.transform(X_test)

    # Predict using logistic regression
    y_predict = model.predict(X_test_maxabs)

    # Flatten full_test_labels, this are the y (true) labels
    y = np.ravel(full_test_labels)

    # Get f1_score and accuracy
    f_score = f1_score(y, y_predict, average='macro')
    accuracy = accuracy_score(y, y_predict)
    return accuracy, f_score


def get_features(full_data, result):
    X_train = np.zeros([len(full_data), result.syn0.shape[1]])
    for i, sentence in enumerate(full_data):
        sum_sentence = np.zeros(result.syn0.shape[1])
        sentence = sentence.split(" ")
        k = len(sentence)
        for w1 in sentence:
            if w1 in result.index2word:
                w_i = 1
                index = result.index2word.index(w1)
                v_i = result.syn0[index]
                sum_sentence += (w_i * v_i) / k
        X_train[i] = sum_sentence

    return X_train


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    # Get train, test files path and output folder full path
    script, train_file_path, test_file_path, output = argv

    # Check if output folder doesn't exist and create it
    if not os.path.exists(output):
        os.makedirs(output)

    # Parse train and tree data using xml.etree.ElementTree
    train_tree = elementTree.parse(codecs.open(train_file_path, 'r+', 'utf-8'))
    test_tree = elementTree.parse(codecs.open(test_file_path, 'r+', 'utf-8'))

    accuracy, f_score = first_question(test_tree, train_tree)

    print('\n' + "classification using BOW model:" + '\n')
    print_scores([['accuracy', accuracy], ['f1-score', f_score]])

    accuracy, f_score = second_question(test_tree, train_tree)
    print('\n' + "classification using embeddings:" + '\n')
    print_scores([['accuracy', accuracy], ['f1-score', f_score]])

    pass


if __name__ == "__main__":
    main()
