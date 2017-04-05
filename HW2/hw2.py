import glob
import os
from collections import Counter
from math import log, sqrt
from sys import argv

number_of_args = 4


def create_file(path, file_name, data):
    """
    Create file with given data
    """
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(os.path.join(path, file_name), 'w+', encoding='utf-8')
    for item in data:
        text = "{}\t\t\t{}{}".format(str(item[0]), str(item[1]), '\n')
        try:
            file.write(text)
        except:
            print("Encountered an error while writing to file")
            print(text)
    file.close()


def load_corpus(path, corpus):
    """
    Load content from path to corpus list
    :param path: given path
    :param corpus: given list
    :return: filled corpus list returned by reference 
    """
    for file_name in glob.glob(os.path.join(path, '*.*')):
        with open(file_name, 'r+', encoding='utf-8') as txt_file:
            # Extend corpus with all lines from current text file
            corpus.extend(txt_file.read().split('\n'))


def get_trigrams(corpus):
    """
    The method separate each line to its tokens and return the given corpus's trigrams
    :param corpus: given corpus
    :return: trigrams in the following structure ("W1" , "W2" , "W3") 
    """
    trigrams = []
    for line in corpus:
        line = line.split(' ')
        trigrams.extend([(line[i], line[i + 1], line[i + 2]) for i in range(len(line)) if i + 2 < len(line)])
    return trigrams


def get_bigrams(corpus):
    """
    he method separate each line to its tokens and return the given corpus's bigrams
    :param corpus: given corpus
    :return: bigrams in the following structure ("W1" , "W2")
    """
    bigrams = []
    for line in corpus:
        line = line.split(' ')
        bigrams.extend(
            [(line[i], line[i + 1]) for i in range(len(line)) if i + 1 < len(line)])
    return bigrams


def get_unigrams(corpus):
    """
    The method returns a list of all tokens
    :param corpus: given corpus
    :return: list of all tokens
    """
    unigrams = []
    for line in corpus:
        line = line.split(' ')
        unigrams.extend([token for token in line])
    return unigrams


def probability(data):
    """
    Calculate given data probability according to the formula from lecture #6
    count(w)/(number of words in text)
    :param data: given data
    :return: probability of each val in data
    """
    data_size = len(data)
    # Unique values in data to prevent more than one probability calculation per value
    unique_data = list(set(data))
    # Count data
    data_counter = Counter(data)
    # Calculate probability of value in data
    data_probability = {}
    for val in unique_data:
        data_probability[val] = data_counter[val] / data_size
    return data_probability


def bigrams_raw_frequency(bigrams, unigrams):
    bigrams_raw_frequency = {}
    unique_bigrams = list(set(bigrams))
    bigrams_count_freq = Counter(bigrams)
    unigrams_size = len(unigrams)

    for bi in unique_bigrams:
        bigrams_raw_frequency[bi] = round((bigrams_count_freq[bi] / unigrams_size) * 1000, 3)
    return bigrams_raw_frequency


def calculate_bigrams_PMI(bigrams_probability, unigrams_probability):
    bigrams_PMI = {}
    for bigram, bigram_probability in bigrams_probability.items():
        w1 = bigram[0]
        w2 = bigram[1]
        # PMI(w1,w2) = log(P(w1w2)/P(w1)*P(w2))
        numerator = bigram_probability
        denominator = unigrams_probability[w1] * unigrams_probability[w2]
        bigrams_PMI[bigram] = round(log(numerator / denominator, 2), 3)
    return bigrams_PMI


def calculate_bigrams_t_test(bigrams_probability, unigrams_probability, unigrams):
    bigrams_T_test = {}
    unigrams_size = len(unigrams)
    for bigram, bigram_probability in bigrams_probability.items():
        w1 = bigram[0]
        w2 = bigram[1]
        numerator = (bigram_probability - (unigrams_probability[w1] * unigrams_probability[w2]))
        denominator = sqrt(bigram_probability / unigrams_size)
        bigrams_T_test[bigram] = round((numerator / denominator), 3)
    return bigrams_T_test


def calculate_X2_test(bigrams_probability, unigrams_probability):
    bigrams_X2_test = {}
    for bigram, bigram_probability in bigrams_probability.items():
        w1 = bigram[0]
        w2 = bigram[1]
        numerator = (bigram_probability - (unigrams_probability[w1] * unigrams_probability[w2]))
        denominator = unigrams_probability[w1] * unigrams_probability[w2]
        bigrams_X2_test[bigram] = round((numerator / denominator), 3)
    return bigrams_X2_test


def calculate_trigrams_T3_test_a(trigrams_probability, unigrams_probability, unigrams):
    trigram_T3_test_a = {}
    unigrams_size = len(unigrams)
    for trigram, trigram_probability in trigrams_probability.items():
        w1 = trigram[0]
        w2 = trigram[1]
        w3 = trigram[2]
        numerator = trigram_probability - unigrams_probability[w1] * unigrams_probability[w2] * unigrams_probability[w3]
        denominator = sqrt(trigram_probability / unigrams_size)
        trigram_T3_test_a[trigram] = round((numerator / denominator), 3)
    return trigram_T3_test_a


def calculate_trigrams_T3_test_b(bigrams_probability, trigrams_probability, unigrams):
    trigram_T3_test_b = {}
    unigrams_size = len(unigrams)
    for trigram, trigram_probability in trigrams_probability.items():
        w1 = trigram[0]
        w2 = trigram[1]
        w3 = trigram[2]
        w12 = (w1, w2)
        w23 = (w2, w3)
        numerator = trigram_probability - bigrams_probability[w12] * bigrams_probability[w23]
        denominator = sqrt(trigram_probability / unigrams_size)
        trigram_T3_test_b[trigram] = round((numerator / denominator), 3)
    return trigram_T3_test_b


def calculate_X3_test_a(trigrams_probability, unigrams_probability):
    trigrams_X3_test_a = {}
    for trigram, trigram_probability in trigrams_probability.items():
        w1 = trigram[0]
        w2 = trigram[1]
        w3 = trigram[2]
        numerator = trigram_probability - unigrams_probability[w1] * unigrams_probability[w2] * unigrams_probability[w3]
        denominator = unigrams_probability[w1] * unigrams_probability[w2] * unigrams_probability[w3]
        trigrams_X3_test_a[trigram] = round((numerator / denominator), 3)
    return trigrams_X3_test_a


def calculate_X3_test_b(bigrams_probability, trigrams_probability):
    trigrams_X3_test_b = {}
    for trigram, trigram_probability in trigrams_probability.items():
        w1 = trigram[0]
        w2 = trigram[1]
        w3 = trigram[2]
        w12 = (w1, w2)
        w23 = (w2, w3)
        numerator = trigram_probability - bigrams_probability[w12] * bigrams_probability[w23]
        denominator = bigrams_probability[w12] * bigrams_probability[w23]
        trigrams_X3_test_b[trigram] = round((numerator / denominator), 3)
    return trigrams_X3_test_b


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    script, first_input_path, second_input_path, output_path = argv
    try:
        # Load 2 given input folder content
        first_corpus = []
        load_corpus(first_input_path, first_corpus)
        second_corpus = []
        load_corpus(second_input_path, second_corpus)
        # Merge the two given corpuses
        merged_corpus = first_corpus + second_corpus

        # Get the corpus's bigrams and unigrams
        bigrams = get_bigrams(merged_corpus)
        unigrams = get_unigrams(merged_corpus)

        # Calculate the bigrams and unigrams probability
        bigrams_probability = probability(bigrams)
        unigrams_probability = probability(unigrams)

        # Calculate bigrams raw frequency
        bigrams_raw_freq = bigrams_raw_frequency(bigrams, unigrams)

        # Calculate bigrams PMI
        bigrams_PMI = calculate_bigrams_PMI(bigrams_probability, unigrams_probability)

        # Calculate bigrams T-test
        bigrams_t_test = calculate_bigrams_t_test(bigrams_probability, unigrams_probability, unigrams)

        # Calculate bigrams X2-test
        bigrams_X2_test = calculate_X2_test(bigrams_probability, unigrams_probability)

        # Get the corpus's trigrams
        trigrams = get_trigrams(merged_corpus)

        # Calculate the trigrams probability
        trigrams_probability = probability(trigrams)

        # Calculate trigrams T3-test a
        trigrams_T3_test_a = calculate_trigrams_T3_test_a(trigrams_probability, unigrams_probability, unigrams)

        # Calculate trigrams T3-test b
        trigrams_T3_test_b = calculate_trigrams_T3_test_b(bigrams_probability, trigrams_probability, unigrams)

        # Calculate trigrams X3-test a
        trigrams_X3_test_a = calculate_X3_test_a(trigrams_probability, unigrams_probability)

        # Calculate trigrams X3-test b
        trigrams_X3_test_a = calculate_X3_test_b(bigrams_probability,trigrams_probability)

    except:
        print("General error")
        exit()


if __name__ == "__main__":
    main()
