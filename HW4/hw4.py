import os
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
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors

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

# Question 2 flags
# Each flag control which section from question 2 to run
# Only one value can be True if both False section 1 is calculated
b_2 = False
c_2 = False
d_2 = False


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


def get_data_from_xml(data):
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

        tokenized_senteces = []
        for sentence in inst[1].getchildren():
            tokens = nltk.word_tokenize(sentence.text.lower())
            tokenized_senteces.extend(tokens)

        if att == CORD:
            cord.append(tokenized_senteces)
        elif att == DIVISION:
            division.append(tokenized_senteces)
        elif att == FORMATION:
            formation.append(tokenized_senteces)
        elif att == PHONE:
            phone.append(tokenized_senteces)
        elif att == PRODUCT:
            product.append(tokenized_senteces)

    return cord, division, formation, phone, product


def print_scores(scores):
    """
    Print scores
    :param scores: given scorres
    :return: 
    """
    for score in scores:
        print(str(score[0]) + ": " + str(round(score[1], 2)) + '\n')


def extract(words):
    """
    analyzer method which define for TfidfVectorizer how to extract from our data
    :param words: given list from our data structure
    :return: expected concatenated string for TfidfVectorizer
    """
    
    # Make sure TfidfVectorizer works on a list of strings
    return words


def first_question(test_tree, train_tree):
    """
    First question: implement logistic regression to recognize "line" category
    :param test_tree: test data , xml structure  
    :param train_tree: train data, xml structure
    :return: 
    """

    # Get train data by classification
    cord, division, formation, phone, product = get_data_from_xml(train_tree)

    # Build train data and labels arrays
    full_labels, full_data = get_full_data([cord, division, formation, phone, product],
                                           [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])

    # Create bag of words using TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer=extract, min_df=1, stop_words='english')
    X = vectorizer.fit_transform(full_data, full_labels)

    # Run logistic regression
    model = LogisticRegression()
    model.fit(X, full_labels)

    # Get test data by classification
    cord, division, formation, phone, product = get_data_from_xml(test_tree)

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

    # Load word2vec embeddings file
    words_embeddings = KeyedVectors.load_word2vec_format("wiki.en.100k.vec", binary=False)

    # Get train data by classification
    cord_train, division_train, formation_train, phone_train, product_train = get_data_from_xml(train_tree)

    # Get test data by classification
    cord_test, division_test, formation_test, phone_test, product_test = get_data_from_xml(test_tree)

    # Build train and test data and labels arrays
    full_train_labels, full_train_data = get_full_data(
        [cord_train, division_train, formation_train, phone_train, product_train],
        [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])

    # Build test data and labels arrays
    full_test_labels, full_test_data = get_full_data(
        [cord_test, division_test, formation_test, phone_test, product_test],
        [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])

    # Get train and test data features by word2vec
    X_full_data = get_features(full_train_data + full_test_data, words_embeddings)

    # Normalize full data features
    scaler = MaxAbsScaler()
    X_full_data_maxabs = scaler.fit_transform(X_full_data)

    # Run logistic regression on normalized train data
    model = LogisticRegression()
    model.fit(X_full_data_maxabs[0: len(full_train_data)], full_train_labels)

    # Predict using logistic regression on normalized test data
    y_predict = model.predict(X_full_data_maxabs[len(full_train_data): len(X_full_data_maxabs)])

    # Flatten full_test_labels, this are the y (true) labels
    y = np.ravel(full_test_labels)

    # Get f1_score and accuracy
    f_score = f1_score(y, y_predict, average='macro')
    accuracy = accuracy_score(y, y_predict)
    return accuracy, f_score


def calculate_weight(x, c=20):
    """
    Calculate weight value
    :param x: given x distance
    :param c: default value 20 
    :return: Calculated weight
    """
    a = 1
    b = 1
    numerator = np.power((x - b), 2)
    denominator = 2 * np.power(c, 2)
    return a * np.exp(-1 * (numerator / denominator))


def get_features(data, words_embeddings):
    """
    Get features according to HW's definition
    b_2: Answer question 2 section b
    c_2: Answer question 2 section c
    d_2: Answer question 2 section d (bonus)
    :param data: given data
    :param words_embeddings: given word2vec embeddings
    :return: 
    """
    X = np.zeros([len(data), words_embeddings.syn0.shape[1]])
    for i, sentence in enumerate(data):
        sum_sentence = np.zeros(words_embeddings.syn0.shape[1])
        k = len(sentence)
        for word in sentence:
            if word in words_embeddings.index2word:
                line_index = 0
                if "line" in sentence:
                    line_index = sentence.index("line")
                x = np.abs(sentence.index(word) - line_index)
                w_i = 1
                if b_2:
                    w_i = calculate_weight(x)
                elif c_2:
                    if sentence.index(word) >= line_index:
                        w_i = calculate_weight(x, 5)
                    else:
                        w_i = calculate_weight(x)
                elif d_2:
                    if np.abs(sentence.index(word) - line_index) <= 4:
                        with open("word.txt", "a+") as text_file:
                            text_file.write("Word is: %s" % word)
                            text_file.write("\n")
                    else:
                        w_i = calculate_weight(x)
                index = words_embeddings.index2word.index(word)
                v_i = words_embeddings.syn0[index]
                sum_sentence += (w_i * v_i) / k
        X[i] = sum_sentence

    return X


def create_file(given_path, bow_scores, word_embeddings_scores):
    """
    Create file with given path and write scores into it
    :param word_embeddings_scores: 
    :param bow_scores: 
    :param given_path: 
    :return: 
    """
    # Make sure the given_path folder exists, if not create it
    drive, path = os.path.splitdrive(given_path)
    path, filename = os.path.split(path)
    folder = os.path.join(drive, path)
    # Check if  folder doesn't exist and create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = open(os.path.join(folder, filename), 'w+', encoding='utf-8')
    try:
        file.write("classification using BOW model:\n")
        file.write("\n")
        for bow_score in bow_scores:
            file.write(str(bow_score[0]) + ": " + str(round(bow_score[1], 2)) + "\n" + "\n")
        file.write("classification using embeddings:\n")
        file.write("\n")
        for w_emb_score in word_embeddings_scores:
            file.write(str(w_emb_score[0]) + ": " + str(round(w_emb_score[1], 2)) + "\n" + "\n")
    except Exception as e:
        print("error while writing to file!")
    file.close()


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    # Get train, test files path and output folder full path
    script, train_file_path, test_file_path, output = argv

    # Parse train and tree data using xml.etree.ElementTree
    train_tree = elementTree.parse(codecs.open(train_file_path, 'r+', 'utf-8'))
    test_tree = elementTree.parse(codecs.open(test_file_path, 'r+', 'utf-8'))

    # First question
    bow_accuracy, bow_f_score = first_question(test_tree, train_tree)

    print('\n' + "classification using BOW model:" + '\n')
    print_scores([['accuracy', bow_accuracy], ['f1-score', bow_f_score]])

    # Second question
    word_embeddings_accuracy, word_embeddings_f_score = second_question(test_tree, train_tree)
    print('\n' + "classification using embeddings:" + '\n')
    print_scores([['accuracy', word_embeddings_accuracy], ['f1-score', word_embeddings_f_score]])

    create_file(output,
                [['accuracy', bow_accuracy], ['f1-score', bow_f_score]],
                [['accuracy', word_embeddings_accuracy], ['f1-score', word_embeddings_f_score]])

    pass


if __name__ == "__main__":
    main()
