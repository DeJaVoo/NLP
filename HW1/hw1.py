import os
import re
from sys import argv, exit

from lxml import html

NON_DIGIT_SPECIAL_CHARTERS = r'(\.|:|\\|/)'

CAPTION_CENTER = 'wp-caption aligncenter'
CAPTION_LEFT = 'wp-caption alignleft'
CAPTION_RIGHT = 'wp-caption alignright'

number_of_args = 3


def get_paragraphs(tree):
    """
    The method purpose is to scrap all <p> elements

    Parameter:
        tree - HTML tree (by lxml structure)
    Returns:
        String of all Paragraphs text
    """
    result = []
    for element in tree:
        # only go deep in divs
        if element.tag == 'div' and element.text_content() != '' and not is_content_class(element):
            res = get_paragraphs(element)  # find paragraphs in nested nodes
            for r in res:
                result.append(r)
        elif is_valid_p_element(element) or element.tag == 'h3':
            p = element.text_content()
            # <p> contains only result, ignoring all scripts, images, etc.
            if p.strip() != '':
                # In case the sentence doesn't contain dot at the end of the string synthetically add dot at the end
                result.append(p)
    return result


def is_valid_p_element(element):
    return element.tag == 'p' and ('class' not in element.attrib or element.attrib['class'] != "wp-caption-text")


def is_content_class(elm):
    clazz = elm.attrib['class']
    is_left_caption = clazz == CAPTION_LEFT
    is_right_caption = clazz == CAPTION_RIGHT
    is_center_caption = clazz == CAPTION_CENTER
    return 'class' in elm.attrib and (is_center_caption or is_right_caption or is_left_caption)


def is_a_digit(c):
    return "0" <= c <= "9"


def is_english_latter(c):
    return "a" <= c <= "z" or "A" <= c <= "Z"


def between_digits(before, after):
    return is_a_digit(before) and is_a_digit(after)


def between_english_letters(before, after):
    return is_english_latter(before) and is_english_latter(after)


def is_sentence_ending_char(c, text, i):
    if c == "." and len(text) > (i + 1) and i > 0:
        before = text[i - 1]
        after = text[i + 1]
        return not between_digits(before, after) and not between_english_letters(before, after)
    return c == "?" or c == "!" or c == "."


def split_into_sentences(text):
    """
    The method purpose is to split the text into a list of sentences
    Parameters:
        text to split
    Returns:
        sentences
    """
    # split on ".", "?", "!"
    sentences_list = []
    sentence = ""
    mode = "during"
    for i, c in enumerate(text):
        if mode == "during":
            sentence += c
            if is_sentence_ending_char(c, text, i):
                mode = "ending"
        else:
            if not is_sentence_ending_char(c, text, i):
                mode = "during"
                sentences_list.append(sentence)
                sentence = c
            else:
                sentence += c
    if sentence != "":
        sentences_list.append(sentence)
    return strip_string_list(sentences_list)


def strip_string_list(sentences_list):
    trimmed_sentences_list = []
    for s in sentences_list:
        trimmed_sentences_list.append(s.strip())
    return trimmed_sentences_list


def create_file_with_given_text(path, file_name, text):
    """
    The method purpose is to create a new file with utf-8 encoding according to given file_name
    and write the given text to file.
    Parameters:
        file_name
        text to save
    Returns:
        None
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        text_file = open(os.path.join(path, file_name), 'w', encoding='utf-8')
        text_file.write(text)
        text_file.close()
    except:
        print("Encountered an error while writing to file")


def tokenize(sentences):
    """
    tokenize the sentences
    """

    newSentences = []
    for s in sentences:
        st = space_around_non_digit_special_characters(s)
        st = space_around_special_characters(st)
        st = space_around_not_middle_of_a_word_special_characters(st)
        st = remove_double_spaces(st)
        if not st.startswith(" "):
            st = " " + st
        if not st.endswith(" "):
            st += " "
        newSentences.append(st)
    return newSentences


def remove_double_spaces(st):
    r = re.compile(r'(\s)+')
    st = r.sub(r' ', st)
    return st


def space_around_special_characters(s):
    """
    Add a whitespace before and after "\" "/" "(" ")" ";" "{"  "}"  "[" "]" "<" ">" "!" "?" "+" "="
    :param s:
    :return:
    """
    r = re.compile(r'(\(|\)|;|\{|\}|\[|\]|<|>|\+|=)')
    return r.sub(r' \1 ', s)
    # return st


def space_around_non_digit_special_characters(s):
    """
    Add a whitespace before and  after "," / "." / ":"  ";" if not surrounded by digits
    :param s:
    :return:
    """
    r = re.compile(r'(?<!\d|\s)' + NON_DIGIT_SPECIAL_CHARTERS + r'(?!\d)')
    st = r.sub(r' \1 ', s)
    r = re.compile(r'(?<!\d)' + NON_DIGIT_SPECIAL_CHARTERS + r'(?!\d|\s)')
    st = r.sub(r' \1 ', st)
    r = re.compile(r'(?<=.)(,)(?!\d\d\d)')  # less then 3 digit
    st = r.sub(r' \1 ', st)
    r = re.compile(r'(?<=.)(,)(?=\d\d\d\d)')  # more then 3 digit
    st = r.sub(r' \1 ', st)
    return st


def space_around_not_middle_of_a_word_special_characters(s):
    """
    Add a whitespace before and after "  if not in the middle of a  hebrew word
    :param s:
    :return:
    """
    # Add a whitespace before " if a whitespace exists after
    r = re.compile(r'(")(?![א-ת])')
    st = r.sub(r' \1 ', s)
    r = re.compile(r'(?<![א-ת])(")')
    st = r.sub(r' \1 ', st)
    r = re.compile(r'(?<![א-ת])(\')(?![a-zA-Z])')
    st = r.sub(r' \1 ', st)
    r = re.compile(r'(?<![א-תa-zA-Z])(\')')
    return r.sub(r' \1 ', st)


def main():
    # Check for expected number of Arguments
    if len(argv) != number_of_args:
        exit("Invalid number of arguments")

    script, url, path = argv
    print("The given path is: " + path)

    try:
        # GET page
        with open(url, "r", encoding='utf-8') as file:
            page = file.read()
        # Get HTML Tree
        tree = html.fromstring(page)
        # Title
        post_entry = '//article[@class="post-entry post"]'
        title = tree.xpath(post_entry + '/h1/text()')[0]

        # Get all article content
        text = get_paragraphs(tree.xpath('//section[@class="post-content "]')[0].getchildren())
        text = [title] + text

        # Step 1: save to file
        create_file_with_given_text(path, "article.txt", "".join(text))

        # Step 2: split into sentences
        sentences = []
        for paragraph in text:
            tmp_sentences = split_into_sentences(paragraph)
            for s in tmp_sentences:
                sentences.append(s)

        create_file_with_given_text(path, "article_sentences.txt", "\n".join(sentences))

        # Step 3: tokenize
        newSentences = tokenize(sentences)
        create_file_with_given_text(path, "article_tokenized.txt", "\n".join(newSentences))

    except ConnectionError as ex:
        print("Connection to geektime failed with ({0}): {1}".format(ex.errno, ex.strerror))
        exit()
    except:
        print("General error")
        exit()


if __name__ == "__main__":
    main()

