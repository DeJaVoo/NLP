import os
from decimal import Decimal
from sys import argv

number_of_args = 4


def read_file(path):
    """
    Load content from path
    :param path: given path
    :return: the file
    """
    result = []
    # for file_name in glob.glob(os.path.join(path, '*.*')):
    with open(path, 'r+', encoding='utf-8') as txt_file:
        # Extend result with all lines from current text file
        result.extend(txt_file.read().split('\n'))
    return result


def write_file(path, text):
    """
    Create file with given path and write text into it
    :param path:
    :param text: a list of lines to write
    :return:
    """
    # Make sure the given_path folder exists, if not create it
    drive, path = os.path.splitdrive(path)
    path, filename = os.path.split(path)
    folder = os.path.join(drive, path)
    # Check if  folder doesn't exist and create it
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    file = open(os.path.join(folder, filename), 'w+', encoding='utf-8')
    try:
        for line in text:
            file.write(line + "\n")
    except Exception as e:
        print("error while writing to file!")
    file.close()


def read_grammar(rules):
    dic = {}
    v = set()
    for rule in rules:
        value = rule.split(" ")
        p = float(value[0])
        res = value[1]
        if value[2] != "->":
            print("error bad pattern")
            continue
        key = (value[3],)
        for val in value[4:]:
            key = key + (val,)
        if key not in dic:
            dic[key] = []
        dic[key].append((res, p))
        v.add(res)
    return dic, v


def cky_algorithm(grammar, v, sentence):
    words = sentence.split(" ")
    n = len(words)
    chart = [[{sym: 0 for sym in v} for j in range(0, n + 1)] for i in range(0, n)]
    symbols = [[{sym: "" for sym in v} for j in range(0, n + 1)] for i in range(0, n)]
    for j in range(1, n + 1):
        word_j = words[j - 1]
        if (word_j,) not in grammar:
            return 0, ""
        for res in grammar[(word_j,)]:
            chart[j - 1][j][res[0]] = res[1]
            symbols[j - 1][j][res[0]] = "[" + res[0] + " " + word_j + "]"

        for i in range(j - 2, -1, -1):
            for k in range(i + 1, j):
                b_list = get_labels(chart, i, k)
                for b in b_list:
                    c_list = get_labels(chart, k, j)
                    for c in c_list:
                        bc = (b, c)
                        if bc in grammar:
                            for value in grammar[bc]:
                                p = value[1] * chart[i][k][b] * chart[k][j][c]
                                if p > chart[i][j][value[0]]:
                                    chart[i][j][value[0]] = p
                                    symbols[i][j][value[0]] = "[" + value[0] + " " + symbols[i][k][b] + symbols[k][j][
                                        c] + "]"
    return chart[0][n]["S"], symbols[0][n]["S"]


def get_labels(chart, i, k):
    x = [b for b in chart[i][k] if chart[i][k][b] > 0]
    return x


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    script, grammar_file, test_sentences_file, output_path = argv
    grammar, v = read_grammar(read_file(grammar_file))
    sentences = read_file(test_sentences_file)
    text = []
    for sentence in sentences:
        text.append(sentence)
        p, sym = cky_algorithm(grammar, v, sentence)
        if p > 0:
            p = '%.2E' % Decimal(p)
            line = p + sym
        else:
            line = "no parse"
        text.append(line)
    write_file(output_path, text)


if __name__ == "__main__":
    main()
