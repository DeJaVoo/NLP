import os
# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
import nltk
from sys import argv
import string
import xml.etree.ElementTree as ET

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
    :param data:
    :param labels:
    :return: 
    """
    full_data = []
    full_labels = []

    for i, data_set in enumerate(data):
        for vec in data_set:
            full_data.append(vec)
            full_labels.append(labels[i])

    return full_labels, full_data


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    # Get train, test files path and output folder full path
    script, train_file_path, test_file_path, output = argv

    # Check if output folder doesn't exist and create it
    if not os.path.exists(output):
        os.makedirs(output)

    train_tree = ET.parse(codecs.open(train_file_path, 'r+', 'utf-8'))
    test_tree = ET.parse(codecs.open(test_file_path, 'r+', 'utf-8'))

    cord = []
    division = []
    formation = []
    phone = []
    product = []
    printable = set(string.printable)
    for inst in train_tree.findall(INSTANCE):

        att = inst[0].attrib[SENSEID]

        for sentence in inst[1].getchildren():

            tokens = nltk.word_tokenize(sentence.text)

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

    # Create bag of words using TfidfVectorizer
    full_labels, full_data = get_full_data([cord, division, formation, phone, product],
                                           [CORD_LBL, DIVISION_LBL, FORMATION_LBL, PHONE_LBL, PRODUCT_LBL])
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    features = vectorizer.fit_transform(full_data, full_labels)

    pass


if __name__ == "__main__":
    main()
