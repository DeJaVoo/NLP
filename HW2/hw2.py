import glob
import operator
import os
from collections import Counter
from math import log, sqrt
from sys import argv

number_of_args = 4


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
    The method separate each line to its tokens and return the given corpus's bigrams
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
    """
    Calculate bigrams raw frequency
    
    raw = bigrams_count_freq/unigrams_size
    :param bigrams: bigrams
    :param unigrams: unigrams
    :return: bigrams raw frequency
    """
    bigrams_raw_frequency = {}
    unique_bigrams = list(set(bigrams))
    bigrams_count_freq = Counter(bigrams)
    unigrams_size = len(unigrams)

    for bigram in unique_bigrams:
        bigrams_raw_frequency[bigram] = round((bigrams_count_freq[bigram] / unigrams_size) * 1000, 3)
    return bigrams_raw_frequency


def calculate_bigrams_PMI(bigrams_probability, unigrams_probability):
    """
    Calculate bigrams PMI
    
    PMI(x,y) = log2(P(xy)/P(x)*P(y))
    :param bigrams_probability: bigrams_probability
    :param unigrams_probability: unigrams_probability
    :return: bigrams PMI
    """
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
    """
    Calculate T test
    
    T = [ P(xy)-P(x)P(y) ] / [sqrt(P(xy)/N)]
    :param bigrams_probability: bigrams_probability
    :param unigrams_probability: unigrams_probability
    :param unigrams: unigrams
    :return: T test
    """
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
    """
    Calculate X2 test
    
    X = [ P(xy)-P(x)P(y) ] / [P(x)P(y)]
    :param bigrams_probability: bigrams_probability
    :param unigrams_probability: unigrams_probability
    :return: X2 test
    """
    bigrams_X2_test = {}
    for bigram, bigram_probability in bigrams_probability.items():
        w1 = bigram[0]
        w2 = bigram[1]
        numerator = (bigram_probability - (unigrams_probability[w1] * unigrams_probability[w2]))
        denominator = unigrams_probability[w1] * unigrams_probability[w2]
        bigrams_X2_test[bigram] = round((numerator / denominator), 3)
    return bigrams_X2_test


def calculate_trigrams_T3_test_a(trigrams_probability, unigrams_probability, unigrams):
    """
    Calculate T3 test a
    
    t3_a =  [ P(xyz)-P(x)P(y)P(z) ] / [sqrt(P(xyz)/N)]
    :param trigrams_probability: trigrams_probability
    :param unigrams_probability: unigrams_probability
    :param unigrams: unigrams
    :return: T3 test a
    """
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
    """
    Calculate T3 test b
    
    t3_b = [ P(xyz)-P(xy)P(yz) ] / [sqrt(P(xyz)/N)]
    :param bigrams_probability: bigrams_probability
    :param trigrams_probability: trigrams_probability
    :param unigrams: unigrams
    :return: T3 test b
    """
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
    """
    Calculate X3 test a
    
    x3_a = [ P(xyz)-P(x)P(y)P(z) ] / [P(x)P(y)p(z)]
    :param unigrams_probability: unigrams probability
    :param trigrams_probability:  trigrams probability
    :return: X3 test a
    """
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
    """
    Calculate X3 test b
    
    x3_b =  [ P(xyz)-P(xy)P(yz) ] / [P(xy)P(yz)]
    :param bigrams_probability: bigrams probability
    :param trigrams_probability:  trigrams probability
    :return: X3 test b
    """
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
        bigrams_T_test = calculate_bigrams_t_test(bigrams_probability, unigrams_probability, unigrams)

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
        trigrams_X3_test_b = calculate_X3_test_b(bigrams_probability, trigrams_probability)

        # Order alphabetically afterwards sort lists by value and take only first 100 elements
        sorted_bigrams_raw_freq = sort_data(bigrams_raw_freq)[:100]
        sorted_bigrams_PMI = sort_data(bigrams_PMI)[:100]
        sorted_bigrams_T_test = sort_data(bigrams_T_test)[:100]
        sorted_bigrams_X2_test = sort_data(bigrams_X2_test)[:100]
        sorted_trigrams_T3_test_a = sort_data(trigrams_T3_test_a)[:100]
        sorted_trigrams_T3_test_b = sort_data(trigrams_T3_test_b)[:100]
        sorted_trigrams_X3_test_a = sort_data(trigrams_X3_test_a)[:100]
        sorted_trigrams_X3_test_b = sort_data(trigrams_X3_test_b)[:100]

        # Save files:
        # freq_raw.txt, pmi_pair.txt, ttest_pair.txt, xtest_pair.txt,
        # ttest_tri_a.txt, ttest_tri_b.txt, xtest_tri_a.txt, xtest_tri_b.txt files
        create_file(output_path, "freq_raw.txt", sorted_bigrams_raw_freq)
        create_file(output_path, "pmi_pair.txt", sorted_bigrams_PMI)
        create_file(output_path, "ttest_pair.txt", sorted_bigrams_T_test)
        create_file(output_path, "xtest_pair.txt", sorted_bigrams_X2_test)
        create_file(output_path, "ttest_tri_a.txt", sorted_trigrams_T3_test_a)
        create_file(output_path, "ttest_tri_b.txt", sorted_trigrams_T3_test_b)
        create_file(output_path, "xtest_tri_a.txt", sorted_trigrams_X3_test_a)
        create_file(output_path, "xtest_tri_b.txt", sorted_trigrams_X3_test_b)

    except:
        print("General error")
        exit()


if __name__ == "__main__":
    main()
