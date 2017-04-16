import glob
import operator
import os
from collections import Counter
from math import log, sqrt
from sys import argv

number_of_args = 4
MIN_COUNT = 20


def create_file(path, file_name, data):
    """
    Create file with given data
    """
    # Make sure the path folder exists, if not create it
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


def sort_data(data):
    """
    The method sort data alphabetic and by value
    :param data: given data
    :return: sorted data
    """
    # Sort given data by alphabetic order
    alphabetic_sorted_list = sorted(data.items(), key=operator.itemgetter(0))
    # Sort given data by value
    sorted_list_by_value = sorted(alphabetic_sorted_list, key=operator.itemgetter(1), reverse=True)
    return sorted_list_by_value


def load_corpus(path):
    """
    Load content from path to corpus list
    :param path: given path
    :return: filled corpus list returned by reference
    """
    corpus = []
    for file_name in glob.glob(os.path.join(path, '*.*')):
        with open(file_name, 'r+', encoding='utf-8') as txt_file:
            # Extend corpus with all lines from current text file
            corpus.extend(txt_file.read().split('\n'))
    return corpus


def get_n_grams(corpus, n):
    """
    The method separate each line to its tokens and return the given corpus's n-grams
    :param corpus: given corpus
    :param n: the number of words in the n-gram
    :return: a list of n-gram tuples
    """
    n_grams = []
    for line in corpus:
        line = split_by_space(line)
        for i in range(len(line) + 1 - n):
            n_gram = tuple()
            for j in range(n):
                n_gram = n_gram + (line[i + j],)
            n_grams.append(n_gram)
    return n_grams


def split_by_space(line):
    line = line.split(' ')
    if line:
        line = list(filter(None, line))  # remove emtpy string
    return line


def get_unigrams(corpus):
    """
    The method returns a list of all tokens
    :param corpus: given corpus
    :return: list of all tokens
    """
    unigrams = []
    for line in corpus:
        line = split_by_space(line)
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


def bigrams_raw_frequency(bigrams, unigrams, words_counter):
    """
    Calculate bigrams raw frequency

    raw = bigrams_count_freq/unigrams_size
    :param words_counter: counter of all the words in the corpus
    :param bigrams: bigrams
    :param unigrams: unigrams
    :return: bigrams raw frequency
    """
    bi_raw_frequency = {}
    unique_bigrams = list(set(bigrams))
    bigrams_count_freq = Counter(bigrams)
    unigrams_size = len(unigrams)
    for bigram in unique_bigrams:
        if valid_word_count(bigram, words_counter):
            bi_raw_frequency[bigram] = round((bigrams_count_freq[bigram] / unigrams_size) * 1000, 3)
    return bi_raw_frequency


def valid_word_count(n_gram, words_counter):
    return all(words_counter[w] >= MIN_COUNT for w in n_gram)


def calculate_bigrams_PMI(bigrams_probability, unigrams_probability, words_counter):
    """
    Calculate bigrams PMI

    PMI(x,y) = log2(P(xy)/P(x)*P(y))
    :param words_counter: counter of all the words in the corpus
    :param bigrams_probability: bigrams_probability
    :param unigrams_probability: unigrams_probability
    :return: bigrams PMI
    """
    bigrams_PMI = {}
    for bigram, bigram_probability in bigrams_probability.items():
        if valid_word_count(bigram, words_counter):
            w1 = bigram[0]
            w2 = bigram[1]
            # PMI(w1,w2) = log(P(w1w2)/P(w1)*P(w2))
            numerator = bigram_probability
            denominator = unigrams_probability[w1] * unigrams_probability[w2]
            bigrams_PMI[bigram] = round(log(numerator / denominator, 2), 3)
    return bigrams_PMI


def calculate_bigrams_t_test(bigrams_probability, unigrams_probability, unigrams, words_counter):
    """
    Calculate T test

    T = [ P(xy)-P(x)P(y) ] / [sqrt(P(xy)/N)]
    :param words_counter: counter of all the words in the corpus
    :param bigrams_probability: bigrams_probability
    :param unigrams_probability: unigrams_probability
    :param unigrams: unigrams
    :return: T test
    """
    bigrams_T_test = {}
    N = len(unigrams)
    for bigram, bigram_probability in bigrams_probability.items():
        if valid_word_count(bigram, words_counter):
            w1 = bigram[0]
            w2 = bigram[1]
            numerator = (bigram_probability - (unigrams_probability[w1] * unigrams_probability[w2]))
            denominator = sqrt(bigram_probability / N)
            bigrams_T_test[bigram] = round((numerator / denominator), 3)
    return bigrams_T_test


def calculate_X2_test(bigrams_probability, unigrams_probability, words_counter):
    """
    Calculate X2 test

    X = [ P(xy)-P(x)P(y) ] / [P(x)P(y)]
    :param words_counter: counter of all the words in the corpus
    :param bigrams_probability: bigrams_probability
    :param unigrams_probability: unigrams_probability
    :return: X2 test
    """
    bigrams_X2_test = {}
    for bigram, bigram_probability in bigrams_probability.items():
        if valid_word_count(bigram, words_counter):
            w1 = bigram[0]
            w2 = bigram[1]
            numerator = (bigram_probability - (unigrams_probability[w1] * unigrams_probability[w2]))
            denominator = unigrams_probability[w1] * unigrams_probability[w2]
            bigrams_X2_test[bigram] = round((numerator / denominator), 3)
    return bigrams_X2_test


def calculate_trigrams_T3_test_a(unigrams, unigrams_probability, trigrams_probability, words_counter):
    """
    Calculate T3 test a

    t3_a =  [ P(xyz)-P(x)P(y)P(z) ] / [sqrt(P(xyz)/N)]
    :param trigrams_probability: trigrams_probability
    :param unigrams_probability: unigrams_probability
    :param unigrams: unigrams
    :param words_counter: counter of all the words in the corpus
    :return: T3 test a
    """
    trigram_T3_test_a = {}
    N = len(unigrams)
    for trigram, trigram_probability in trigrams_probability.items():
        if valid_word_count(trigram, words_counter):
            w1 = trigram[0]
            w2 = trigram[1]
            w3 = trigram[2]
            numerator = trigram_probability - unigrams_probability[w1] * unigrams_probability[w2] * \
                                              unigrams_probability[w3]
            denominator = sqrt(trigram_probability / N)
            trigram_T3_test_a[trigram] = round((numerator / denominator), 3)
    return trigram_T3_test_a


def calculate_trigrams_T3_test_b(unigrams, bigrams_probability, trigrams_probability, words_counter):
    """
    Calculate T3 test b

    t3_b = [ P(xyz)-P(xy)P(yz) ] / [sqrt(P(xyz)/N)]
    :param bigrams_probability: bigrams_probability
    :param trigrams_probability: trigrams_probability
    :param unigrams: unigrams
    :param words_counter: counter of all the words in the corpus
    :return: T3 test b
    """
    trigram_T3_test_b = {}
    N = len(unigrams)
    for trigram, trigram_probability in trigrams_probability.items():
        if valid_word_count(trigram, words_counter):
            w1 = trigram[0]
            w2 = trigram[1]
            w3 = trigram[2]
            w12 = (w1, w2)
            w23 = (w2, w3)
            numerator = trigram_probability - bigrams_probability[w12] * bigrams_probability[w23]
            denominator = sqrt(trigram_probability / N)
            trigram_T3_test_b[trigram] = round((numerator / denominator), 3)
    return trigram_T3_test_b


def calculate_X3_test_a(unigrams_probability, trigrams_probability, words_counter):
    """
    Calculate X3 test a

    x3_a = [ P(xyz)-P(x)P(y)P(z) ] / [P(x)P(y)p(z)]
    :param unigrams_probability: unigrams probability
    :param trigrams_probability:  trigrams probability
    :param words_counter: counter of all the words in the corpus
    :return: X3 test a
    """
    trigrams_X3_test_a = {}
    for trigram, trigram_probability in trigrams_probability.items():
        if valid_word_count(trigram, words_counter):
            w1 = trigram[0]
            w2 = trigram[1]
            w3 = trigram[2]
            numerator = trigram_probability - unigrams_probability[w1] * unigrams_probability[w2] * \
                                              unigrams_probability[w3]
            denominator = unigrams_probability[w1] * unigrams_probability[w2] * unigrams_probability[w3]
            trigrams_X3_test_a[trigram] = round((numerator / denominator), 3)
    return trigrams_X3_test_a


def calculate_X3_test_b(bigrams_probability, trigrams_probability, words_counter):
    """
    Calculate X3 test b

    x3_b =  [ P(xyz)-P(xy)P(yz) ] / [P(xy)P(yz)]
    :param bigrams_probability: bigrams probability
    :param trigrams_probability:  trigrams probability
    :param words_counter: counter of all the words in the corpus
    :return: X3 test b
    """
    trigrams_X3_test_b = {}
    for trigram, trigram_probability in trigrams_probability.items():
        if valid_word_count(trigram, words_counter):
            w1 = trigram[0]
            w2 = trigram[1]
            w3 = trigram[2]
            w12 = (w1, w2)
            w23 = (w2, w3)
            numerator = trigram_probability - bigrams_probability[w12] * bigrams_probability[w23]
            denominator = bigrams_probability[w12] * bigrams_probability[w23]
            trigrams_X3_test_b[trigram] = round((numerator / denominator), 3)
    return trigrams_X3_test_b


def get_corpus_data(corpus):
    unigram = get_unigrams(corpus)
    unigrams_counter = Counter(unigram)
    merged_bigrams = get_n_grams(corpus, 2)
    trigrams = get_n_grams(corpus, 3)
    return merged_bigrams, unigram, trigrams, unigrams_counter


def get_trigrams_results(bigram_probability, trigram_probability, unigram_probability, unigrams, unigrams_counter):
    # Calculate trigrams T3-test a
    trigrams_T3_test_a = calculate_trigrams_T3_test_a(unigrams, unigram_probability, trigram_probability,
                                                      unigrams_counter)
    # Calculate trigrams T3-test b
    trigrams_T3_test_b = calculate_trigrams_T3_test_b(unigrams, bigram_probability, trigram_probability,
                                                      unigrams_counter)
    # Calculate trigrams X3-test a
    trigrams_X3_test_a = calculate_X3_test_a(unigram_probability, trigram_probability, unigrams_counter)
    # Calculate trigrams X3-test b
    trigrams_X3_test_b = calculate_X3_test_b(bigram_probability, trigram_probability, unigrams_counter)
    sorted_trigrams_T3_test_a = sort_data(trigrams_T3_test_a)[:10000]
    sorted_trigrams_T3_test_b = sort_data(trigrams_T3_test_b)[:10000]
    sorted_trigrams_X3_test_a = sort_data(trigrams_X3_test_a)[:10000]
    sorted_trigrams_X3_test_b = sort_data(trigrams_X3_test_b)[:10000]
    return sorted_trigrams_T3_test_a, sorted_trigrams_T3_test_b, sorted_trigrams_X3_test_a, sorted_trigrams_X3_test_b


def intersect(a, b):
    res = [v1 for v1 in a if any(v1[0] == v2[0] for v2 in b)]
    return res


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    script, first_input_path, second_input_path, output_path = argv
    try:
        # Load 2 given input folder content
        first_corpus = load_corpus(first_input_path)
        second_corpus = load_corpus(second_input_path)
        # Merge the two given corpuses
        merged_corpus = first_corpus + second_corpus

        # Get the corpus's bigrams and unigrams and trigrams
        merged_bigrams, merged_unigrams, merged_trigrams, merged_unigrams_counter = get_corpus_data(merged_corpus)
        first_bigrams, first_unigrams, first_trigrams, first_unigrams_counter = get_corpus_data(first_corpus)
        second_bigrams, second_unigrams, second_trigrams, second_unigrams_counter = get_corpus_data(second_corpus)

        # Calculate the bigrams and unigrams probability
        unigrams_probability = probability(merged_unigrams)
        bigrams_probability = probability(merged_bigrams)

        # Calculate the trigrams probability
        first_uni_probability = probability(first_unigrams)
        first_bi_probability = probability(first_bigrams)
        first_tri_probability = probability(first_trigrams)
        second_uni_probability = probability(second_unigrams)
        second_bi_probability = probability(second_bigrams)
        second_tri_probability = probability(second_trigrams)

        # Calculate bigrams raw frequency
        bigrams_raw_freq = bigrams_raw_frequency(merged_bigrams, merged_unigrams, merged_unigrams_counter)

        # Calculate bigrams PMI
        bigrams_PMI = calculate_bigrams_PMI(bigrams_probability, unigrams_probability, merged_unigrams_counter)

        # Calculate bigrams T-test
        bigrams_T_test = calculate_bigrams_t_test(bigrams_probability, unigrams_probability, merged_unigrams,
                                                  merged_unigrams_counter)

        # Calculate bigrams X2-test
        bigrams_X2_test = calculate_X2_test(bigrams_probability, unigrams_probability, merged_unigrams_counter)

        # Order alphabetically afterwards sort lists by value and take only first 100 elements
        sorted_bigrams_raw_freq = sort_data(bigrams_raw_freq)[:100]
        sorted_bigrams_PMI = sort_data(bigrams_PMI)[:100]
        sorted_bigrams_T_test = sort_data(bigrams_T_test)[:100]
        sorted_bigrams_X2_test = sort_data(bigrams_X2_test)[:100]

        first_T3_test_a, first_T3_test_b, first_X3_test_a, first_X3_test_b = \
            get_trigrams_results(first_bi_probability, first_tri_probability, first_uni_probability, first_unigrams,
                                 first_unigrams_counter)
        second_T3_test_a, second_T3_test_b, second_X3_test_a, second_X3_test_b = \
            get_trigrams_results(second_bi_probability, second_tri_probability, second_uni_probability, second_unigrams,
                                 second_unigrams_counter)

        # return all(words_counter[w] >= MIN_COUNT for w in n_gram)

        t3_test_a = intersect(first_T3_test_a, second_T3_test_a)
        t3_test_b = intersect(first_T3_test_b, second_T3_test_b)
        x3_test_a = intersect(first_X3_test_a, second_X3_test_a)
        x3_test_b = intersect(first_X3_test_b, second_X3_test_b)

        # Save files:
        # freq_raw.txt, pmi_pair.txt, ttest_pair.txt, xtest_pair.txt,
        # ttest_tri_a.txt, ttest_tri_b.txt, xtest_tri_a.txt, xtest_tri_b.txt files
        create_file(output_path, "freq_raw.txt", sorted_bigrams_raw_freq)
        create_file(output_path, "pmi_pair.txt", sorted_bigrams_PMI)
        create_file(output_path, "ttest_pair.txt", sorted_bigrams_T_test)
        create_file(output_path, "xtest_pair.txt", sorted_bigrams_X2_test)
        create_file(output_path, "ttest_tri_a.txt", t3_test_a)
        create_file(output_path, "ttest_tri_b.txt", t3_test_b)
        create_file(output_path, "xtest_tri_a.txt", x3_test_a)
        create_file(output_path, "xtest_tri_b.txt", x3_test_b)

    except:
        print("General error")
        exit()


if __name__ == "__main__":
    main()
