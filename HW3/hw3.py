import csv
import numpy as np
from random import shuffle
from sys import argv

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest

BRITNEY_LBL = 1

BEATLES_LBL = 0

number_of_args = 4


def read_csv_by_filter(path, filter_value):
    """
    Read csv file with given filter value
    :param path: path to file
    :param filter_value: string to filter list (artist name)
    :return: list of all value in row[1] (songs)
    """
    with open(path, 'r', encoding='utf8') as csv_file:
        data = []
        for row in csv.reader(csv_file, delimiter=','):
            if row[3] == filter_value:
                data.append(row[1])
    return data


def load_words(path):
    """
    Read list of words from file
    :param path: given file path 
    :return: list of words
    """
    words = list()
    with open(path, 'r+', encoding='utf-8') as file:
        for line in file:
            s = line.strip().replace(",", "").replace("\'", "").replace("\"", "")
            words.append(s)
    return words


def build_feature_vector(path, songs):
    """
    
    :param path: 
    :param songs: 
    :return: 
    """
    words = load_words(path)
    words_length = len(words)
    vectors = []
    for j, song in enumerate(songs):
        vector = [0] * words_length
        # s = song.split("-")
        # for i in range(words_length):
        #     if words[i] in s:
        #         vector[i] = 1
        for i in range(words_length):
            vector[i] = count_occurrences(words[i], song)
        vectors.append([vector, song])
    return vectors


def count_occurrences(word, sentence):
    return sentence.split("-").count(word)


def ten_fold_cross_validation(classifier, data, target):
    """
    Get 10 fold cross validation and return average
    :param data: 
    :param classifier: 
    :return: 
    """
    scores = cross_val_score(classifier, data, target, cv=10)
    avg_score = sum(scores, 0.0) / len(scores)
    return avg_score


def print_scores(scores):
    for score in scores:
        print("- " + str(score[0]) + ": " + str(score[1]))


def get_classifiers_scores(data, target):
    """
    get classifiers scores
    :param data: training data
    :param target: target data 
    :return: 
    """
    # SVM
    svm_classifier = svm.SVC()
    svm_score = ten_fold_cross_validation(svm_classifier, data, target)

    # NB
    nb_classifier = MultinomialNB()
    nb_classifier.fit(data, target)
    MultinomialNB()
    nb_score = ten_fold_cross_validation(nb_classifier, data, target)

    # Decision Tree
    decision_tree_classifier = DecisionTreeClassifier()
    dt_score = ten_fold_cross_validation(decision_tree_classifier, data, target)

    # KNN
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(data, target)
    knn_score = ten_fold_cross_validation(knn_classifier, data, target)

    return [['SVM', svm_score], ['Naive Bayes', nb_score], ['DecisionTree', dt_score], ['KNN', knn_score]]


def create_evaluation_data(data, labels):
    """
    Generate test_data, training_data from given data
    :param data: [first_data , second_data]
    :param labels: data labels [first_lbl, second_lbl]
    :return: test_data, training_data
    """
    # Build first class vectors
    first_data = data[0]
    second_data = data[1]

    db = []
    for i in range(len(first_data)):
        vec = first_data[i][0]
        db.append([vec, labels[0]])
    # Build second class vectors
    for i in range(len(second_data)):
        vec = second_data[i][0]
        db.append([vec, labels[1]])
    shuffle(db)

    training_data = [i[0] for i in db]
    test_data = [i[1] for i in db]
    return test_data, training_data


def get_full_data(first_data, second_data, labels):
    data = []
    targets = []

    # Build Positive Vectors
    for i in range(len(first_data)):
        data.append(first_data[i])
        targets.append(labels[0])
    # Build Negative Vectors
    for i in range(len(second_data)):
        data.append(second_data[i])
        targets.append(labels[1])

    return data, targets


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    script, input_folder, words_file_input_path, best_words_output_path = argv

    # Read all songs per given artist
    beatles_songs = read_csv_by_filter(input_folder, 'beatles')
    britney_spears_songs = read_csv_by_filter(input_folder, 'britney-spears')
    beatles_features = build_feature_vector(words_file_input_path, beatles_songs)
    britney_spears_features = build_feature_vector(words_file_input_path, britney_spears_songs)

    # Step 1: top frequent words features
    test_data, training_data = create_evaluation_data([beatles_features, britney_spears_features], [BEATLES_LBL,
                                                                                                    BRITNEY_LBL])
    print('\n' + "step1 (top frequent words features):" + '\n')
    print_scores(get_classifiers_scores(training_data, test_data))

    # Step 2: bag of words
    (full_data, full_labels) = get_full_data(beatles_songs, britney_spears_songs , [BEATLES_LBL, BRITNEY_LBL])
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    features = vectorizer.fit_transform(full_data, full_labels)
    vectors = features.A
    print('\n' + "step2 (bag-of-words):" + '\n')
    print_scores(get_classifiers_scores(vectors, test_data))

    #Step 3:selected best features
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    features = vectorizer.fit_transform(full_data)
    vectors = features.A
    all_names = vectorizer.get_feature_names()
    # take 50 words
    b = SelectKBest(k=50)
    b.fit_transform(vectors, test_data)
    indices = b.get_support(indices="true")
    names = np.array(all_names)[indices]
    # Save To File
    file = open(best_words_output_path, 'w+', encoding='utf-8')
    for item in names:
        try:
            file.write(item + '\n')
        except Exception as e:
            print("error while writing to file!")
    file.close()

    print('\n' + "step3 (selected best features):" + '\n')


if __name__ == "__main__":
    main()
