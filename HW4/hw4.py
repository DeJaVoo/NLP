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
from sklearn.feature_selection import SelectKBest

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

labels_dic = {CORD: CORD_LBL, DIVISION: DIVISION_LBL, FORMATION: FORMATION_LBL, PHONE: PHONE_LBL, PRODUCT: PRODUCT_LBL}

number_of_args = 4

# Question 2 flags
# Each flag control which section of question 2 to run
# Only one value can be True
# if all values False question 2 sub-section a is calculated
b_2 = False
c_2 = True
d_2 = False


def tokenize_sentences(inst):
    """
    tokenize sentences
    :param inst: 
    :return: 
    """
    result = []
    for sentence in inst[1].getchildren():
        tokens = nltk.word_tokenize(sentence.text.lower())
        result.extend(tokens)
    return result


def get_full_data_from_xml(data):
    """
    Get full data from xml
    :param data: given xml tree
    :return: 
    """
    full_data = []
    full_labels = []

    for inst in data.findall(INSTANCE):
        att = inst[0].attrib[SENSEID]
        full_data.append(tokenize_sentences(inst))
        full_labels.append(labels_dic[att])

    return full_labels, full_data


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


def first_question(train_data, train_labels, test_data, test_labels):
    """
    First question: implement logistic regression to recognize "line" category
    :param train_data: the train data
    :param train_labels: the train labels
    :param test_data: the test data
    :param test_labels: the test labels
    :return:
    """

    # Create bag of words using TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer=extract, min_df=1, stop_words='english')
    X = vectorizer.fit_transform(train_data, train_labels)

    # Run logistic regression
    model = LogisticRegression()
    model.fit(X, train_labels)

    # Trasnform test data to same shape of train data
    X_test = vectorizer.transform(test_data, test_labels)

    # Predict using logistic regression
    y_predict = model.predict(X_test)

    # Flatten full_test_labels, this are the y (true) labels
    y = np.ravel(test_labels)

    # Get f1_score and accuracy
    f_score = f1_score(y, y_predict, average='macro')
    accuracy = accuracy_score(y, y_predict)
    return accuracy, f_score


def second_question(train_data, train_labels, test_data, test_labels):
    """
    Second question:
    :param train_data: the train data
    :param train_labels: the train labels
    :param test_data: the test data
    :param test_labels: the test labels
    :return:
    """
    # prevent wrong flags values
    if (b_2 and (c_2 or d_2)) or (not b_2 and c_2 and d_2):
        raise ValueError('Question 2: you can\'t set more than one value as True for question 2 flags')

    # Load word2vec embeddings file
    words_embeddings = KeyedVectors.load_word2vec_format("wiki.en.100k.vec", binary=False)

    # Get train and test data features by word2vec
    X_full_data = get_features(train_data + test_data, train_labels + test_labels, words_embeddings)

    # Normalize full data features
    scaler = MaxAbsScaler()
    X_full_data_maxabs = scaler.fit_transform(X_full_data)

    # Run logistic regression on normalized train data
    model = LogisticRegression()
    model.fit(X_full_data_maxabs[0: len(train_data)], train_labels)

    # Predict using logistic regression on normalized test data
    y_predict = model.predict(X_full_data_maxabs[len(train_data): len(X_full_data_maxabs)])

    # Flatten full_test_labels, this are the y (true) labels
    y = np.ravel(test_labels)

    # Get f1_score and accuracy
    f_score = f1_score(y, y_predict, average='macro')
    accuracy = accuracy_score(y, y_predict)
    return accuracy, f_score


def bonus_section(data, labels, k):
    """
    The purpose of this method is to calculate to top K importance words
    :param data: 
    :param labels: 
    :param k: The number of words to select
    :return: top K words
    """
    if d_2:
        vectorizer = TfidfVectorizer(analyzer=extract, min_df=1, stop_words='english')
        features = vectorizer.fit_transform(data)
        vectors = features.A
        features_names = vectorizer.get_feature_names()
        # take k words
        select_k_best = SelectKBest(k=k)
        select_k_best.fit_transform(vectors, labels)
        indices = select_k_best.get_support(indices="true")
        return np.array(features_names)[indices]
    else:
        return []


def calculate_gaussian(x, c):
    """
    Calculate weight value
    :param x: given x distance
    :param c: given c
    :return: Calculated weight
    """
    a = 1
    b = 1
    numerator = np.power((x - b), 2)
    denominator = 2 * np.power(c, 2)
    return a * np.exp(-1 * (numerator / denominator))


def calculate_weight(line_index, names, sentence, word, x):
    w_i = 1
    if b_2:
        w_i = calculate_gaussian(x, 20)
    elif c_2:
        if sentence.index(word) >= line_index:
            w_i = calculate_gaussian(x, 5)
        else:
            w_i = calculate_gaussian(x, 20)
    elif d_2:
        distance = np.abs(sentence.index(word) - line_index)
        if word in names:
            w_i = calculate_gaussian(x, 40)
        elif sentence.index(word) >= line_index:
            if distance <= 3:
                w_i = calculate_gaussian(x, 25)
            else:
                w_i = calculate_gaussian(x, 2)
        else:
            w_i = calculate_gaussian(x, 20)
    return w_i


def get_features(data, labels, words_embeddings):
    """
    Get features according to HW's definition
    b_2: Answer question 2 section b
    c_2: Answer question 2 section c
    d_2: Answer question 2 section d (bonus)
    :param data: given data
    :param labels: given labels for the data
    :param words_embeddings: given word2vec embeddings
    :return:
    """
    # Question 2 sub-section d (bonus) will only run if d_2 flag is set to True
    names = bonus_section(data, labels, 50)

    X = np.zeros([len(data), words_embeddings.syn0.shape[1]])
    for i, sentence in enumerate(data):
        sum_sentence = np.zeros(words_embeddings.syn0.shape[1])
        k = len(sentence)
        for word in sentence:
            line_index = get_line_index(sentence)
            if word in words_embeddings.index2word:
                x = np.abs(sentence.index(word) - line_index)
                # weight default value is 1 (Question 2 sub-section a)
                w_i = calculate_weight(line_index, names, sentence, word, x)
                index = words_embeddings.index2word.index(word)
                v_i = words_embeddings.syn0[index]
                sum_sentence += (w_i * v_i) / k
        X[i] = sum_sentence

    return X


def get_line_index(sentence):
    line_index = 0
    if "line" in sentence:
        line_index = sentence.index("line")
    elif "lines" in sentence:
        line_index = sentence.index("lines")
    return line_index


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
    if folder != '' and not os.path.exists(folder):
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

    train_labels, train_data = get_full_data_from_xml(train_tree)
    test_labels, test_data = get_full_data_from_xml(test_tree)

    # First question
    bow_accuracy, bow_f_score = first_question(train_data, train_labels, test_data, test_labels)

    print('\n' + "classification using BOW model:" + '\n')
    print_scores([['accuracy', bow_accuracy], ['f1-score', bow_f_score]])

    # Second question
    word_embeddings_accuracy, word_embeddings_f_score = second_question(train_data, train_labels, test_data,
                                                                        test_labels)
    print('\n' + "classification using embeddings:" + '\n')
    print_scores([['accuracy', word_embeddings_accuracy], ['f1-score', word_embeddings_f_score]])

    create_file(output,
                [['accuracy', bow_accuracy], ['f1-score', bow_f_score]],
                [['accuracy', word_embeddings_accuracy], ['f1-score', word_embeddings_f_score]])

    pass


if __name__ == "__main__":
    main()
