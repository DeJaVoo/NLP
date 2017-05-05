import csv
import os
from sys import argv

import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

BRITNEY_LBL = 1

BEATLES_LBL = 0

number_of_args = 4


def read_csv_by_filter(path, filter_value):
    """
    Read csv file with given filter value
    :param path: path to file
    :param filter_value: string to filter list (artist name)
    :return: list of all value in row[5] (lyrics)
    """
    with open(path, 'r', encoding='utf8') as csv_file:
        data = []
        for row in csv.reader(csv_file, delimiter=','):
            if row[3] == filter_value:
                if row[5]:
                    data.append(row[5])
    return data


def get_tokens(lyrics):
    """
    Get token of given lyrics
    :param lyrics: list of strings
    :return: 
    """
    tokens = []
    lines = lyrics.split("\n")

    for line in lines:
        # list of tokens
        line = line.lower().replace(",", "").replace("\'", "").replace("\"", "")
        line = line.split(' ')
        # get all unigrams (just list of tokens)
        tokens.extend([token for token in line])

    return tokens


def load_words(path):
    """
    Read list of words from file
    :param path: given file path 
    :return: list of words
    """
    words = list()
    with open(path, 'r+', encoding='utf-8') as file:
        for line in file:
            s = line.strip().lower().replace(",", "").replace("\'", "").replace("\"", "")
            words.append(s)
    return words


def build_feature_vector(path, songs):
    """
    build feature vector
    :param path: given words path
    :param songs: given songs DB
    :return: 
    """
    words = load_words(path)
    words_length = len(words)
    vectors = []
    for j, song in enumerate(songs):
        vector = [0] * words_length
        song = get_tokens(song)
        # check if given word is in song name, if yes mark 1 in i index
        for i in range(words_length):
            if words[i] in song:
                vector[i] = 1
        # save the number of occurrences of a given word in song name
        # for i in range(words_length):
        #     vector[i] = count_occurrences(words[i], song)
        vectors.append([vector, j])
    return vectors


def count_occurrences(word, sentence):
    return sentence.count(word)


def ten_fold_cross_validation(classifier, data, test):
    """
    Get 10 fold cross validation and return average
    :param test:  test data
    :param data: training data
    :param classifier: 
    :return: 
    """
    scores = cross_val_score(classifier, data, test, cv=10)
    avg_score = sum(scores, 0.0) / len(scores)
    return avg_score


def print_scores(scores):
    for score in scores:
        print("- " + str(score[0]) + ": " + str(round(score[1],2)))


def get_classifiers_scores(data, test):
    """
    get classifiers scores
    :param data: training data
    :param test: test data 
    :return: 
    """
    # SVM
    svm_classifier = svm.SVC()
    svm_score = ten_fold_cross_validation(svm_classifier, data, test)

    # NB
    nb_classifier = MultinomialNB()
    nb_classifier.fit(data, test)
    MultinomialNB()
    nb_score = ten_fold_cross_validation(nb_classifier, data, test)

    # Decision Tree
    decision_tree_classifier = DecisionTreeClassifier()
    dt_score = ten_fold_cross_validation(decision_tree_classifier, data, test)

    # KNN
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(data, test)
    knn_score = ten_fold_cross_validation(knn_classifier, data, test)

    return [['SVM', svm_score], ['Naive Bayes', nb_score], ['DecisionTree', dt_score], ['KNN', knn_score]]


def create_evaluation_data(data, labels):
    """
    Generate test_data, training_data from given data
    :param data: [first_data , second_data]
    :param labels: data labels [first_lbl, second_lbl]
    :return: test_data, training_data
    """

    first_data = data[0]
    second_data = data[1]

    db = []
    # Build first class vectors
    for i in range(len(first_data)):
        vec = first_data[i][0]
        db.append([vec, labels[0]])
    # Build second class vectors
    for i in range(len(second_data)):
        vec = second_data[i][0]
        db.append([vec, labels[1]])

    training_data = [i[0] for i in db]
    test_data = [i[1] for i in db]
    return test_data, training_data


def get_full_data(first_data, second_data, labels):
    """
    Return the 2 lists, one list contains both given datasets (all DB)
    second list contains all the labels
    :param first_data: 
    :param second_data: 
    :param labels: 
    :return: 
    """
    full_data = []
    full_labels = []

    # Build first label vectors
    for i in range(len(first_data)):
        full_data.append(first_data[i])
        full_labels.append(labels[0])
    # Build second label Vectors
    for i in range(len(second_data)):
        full_data.append(second_data[i])
        full_labels.append(labels[1])

    return full_data, full_labels


def create_file(given_path, names):
    """
    Create file with given path
    :param given_path: 
    :param names: 
    :return: 
    """
    # Make sure the given_path folder exists, if not create it
    drive, path = os.path.splitdrive(given_path)
    path, filename = os.path.split(path)
    folder = os.path.join(drive, path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = open(os.path.join(folder, filename), 'w+', encoding='utf-8')
    for item in names:
        try:
            file.write(item + '\n')
        except Exception as e:
            print("error while writing to file!")
    file.close()


def first_question(beatles_songs, britney_spears_songs, words_file_input_path):
    """
    Answer the first question
    :param beatles_songs: 
    :param britney_spears_songs: 
    :param words_file_input_path: 
    :return: 
    """
    beatles_features = build_feature_vector(words_file_input_path, beatles_songs)
    britney_spears_features = build_feature_vector(words_file_input_path, britney_spears_songs)
    test_data, training_data = create_evaluation_data([beatles_features, britney_spears_features], [BEATLES_LBL,
                                                                                                    BRITNEY_LBL])
    return test_data, training_data


def second_question(beatles_songs, britney_spears_songs):
    """
    Answer the second question
    :param beatles_songs: 
    :param britney_spears_songs: 
    :return: 
    """
    (full_data, full_labels) = get_full_data(beatles_songs, britney_spears_songs, [BEATLES_LBL, BRITNEY_LBL])
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    features = vectorizer.fit_transform(full_data, full_labels)
    vectors = features.A
    return full_data, full_labels, vectors


def third_question(best_words_output_path, full_data, full_labels):
    """
    Answer the third question
    :param best_words_output_path: 
    :param full_data: 
    :param full_labels: 
    :return: 
    """
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    features = vectorizer.fit_transform(full_data)
    vectors = features.A
    features_names = vectorizer.get_feature_names()
    # take 50 words
    select_50_best = SelectKBest(k=50)
    select_50_best.fit_transform(vectors, full_labels)
    indices = select_50_best.get_support(indices="true")
    names = np.array(features_names)[indices]
    create_file(best_words_output_path, names)
    return names


def fourth_question(full_data, names):
    """
    Answer the fourth question
    :param full_data: 
    :param names: 
    :return: 
    """
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', vocabulary=names)
    features = vectorizer.fit_transform(full_data)
    vectors = features.A
    return vectors


def fifth_question(input_folder):
    dean_martin_songs = read_csv_by_filter(input_folder, 'dean-martin')
    albert_hammond_songs = read_csv_by_filter(input_folder, 'albert-hammond')
    full_data, full_labels = get_full_data(dean_martin_songs, albert_hammond_songs, [0, 1])
    # take 50 words
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    features = vectorizer.fit_transform(full_data)
    vectors = features.A
    features_names = vectorizer.get_feature_names()
    # take 50 words
    select_50_best = SelectKBest(k=50)
    select_50_best.fit_transform(vectors, full_labels)
    indices = select_50_best.get_support(indices="true")
    names = np.array(features_names)[indices]

    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', vocabulary=names)
    features = vectorizer.fit_transform(full_data)
    vectors = features.A
    return vectors, full_labels


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    script, input_folder, words_file_input_path, best_words_output_path = argv

    # Read all songs per given artist
    beatles_songs = read_csv_by_filter(input_folder, 'beatles')
    britney_spears_songs = read_csv_by_filter(input_folder, 'britney-spears')

    # Step 1: top frequent words features
    test_data, training_data = first_question(beatles_songs, britney_spears_songs, words_file_input_path)
    print('\n' + "step1 (top frequent words features):" + '\n')
    print_scores(get_classifiers_scores(training_data, test_data))

    # Step 2: bag of words
    full_data, full_labels, vectors = second_question(beatles_songs, britney_spears_songs)
    print('\n' + "step2 (bag-of-words):" + '\n')
    print_scores(get_classifiers_scores(vectors, full_labels))

    # Step 3: choose 50 most meaningful words
    names = third_question(best_words_output_path, full_data, full_labels)

    # Step 4: selected best features
    vectors = fourth_question(full_data, names)
    print('\n' + "step4 (selected best features):" + '\n')
    print_scores(get_classifiers_scores(vectors, full_labels))

    # Step 5: new classification
    # Read all songs per given artist
    vectors, full_labels = fifth_question(input_folder)
    print('\n' + "step5 (new classification):" + '\n')
    print_scores(get_classifiers_scores(vectors, full_labels))


if __name__ == "__main__":
    main()
