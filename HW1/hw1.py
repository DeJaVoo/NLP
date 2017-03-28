import os
import re
from sys import argv, exit

from lxml import html

NON_DIGIT_SPECIAL_CHARTERS = r'(\.|:)'

SPECIAL_CHARACTERS_GROUP = r'(\\|/|\(|\)|;|\{|\}|\[|\]|<|>|!|\?|\+|=)'

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


def is_in_of_sentence_char(c):
    return c == "?" or c == "." or c == "!"


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
    for c in text:
        if mode == "during":
            sentence += c
            if is_in_of_sentence_char(c):
                mode = "ending"
        else:
            if not is_in_of_sentence_char(c):
                mode = "during"
                sentences_list.append(sentence)
                sentence = c
            else:
                sentence += c
    if sentence != "":
        sentences_list.append(sentence)
    return strip_string_list(sentences_list)
    # sentences_endings = re.compile(r'[^\.!?]*[\.!?\n]+', re.DOTALL)
    # return re.findall(sentences_endings, text)


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
    r = re.compile(r'(\\|/|\(|\)|;|\{|\}|\[|\]|<|>|!|\?|\+|=)')
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
    Add a whitespace before and after " ' if not in the middle of a word
    :param s:
    :return:
    """
    # Add a whitespace before " if a whitespace exists after
    r = re.compile(r'(")(?![א-ת])')
    st = r.sub(r' \1 ', s)
    r = re.compile(r'(?<![א-ת])(")')
    return r.sub(r' \1 ', st)


def tokenize2(sentences):
    """
    tokenize the sentences
    """

    newSentences = []
    for s in sentences:
        # Add a whitespace before "," / "." / ":" ( not surrounded by digits )
        r = re.compile(r'(?<!\d|\s)(,|\.|:|;)(?!=\d)')
        s1 = (r.sub(r' \1', s))
        r = re.compile(r'(?<!\s)(\\|/|\(|\)|;|\{|\}|\[|\]|<|>|!|\?|\+|=)')
        s2 = (r.sub(r' \1', s1))
        # Add a whitespace after "," / "." / ":" ( not surrounded by digits )
        r = re.compile(r'(?<!\d)(,|\.|:)(?!=\d|\s)')
        s3 = (r.sub(r'\1 ', s2))
        r = re.compile(r'(\\|/|\(|\)|;|\{|\}|\[|\]|<|>|!|\?|\+|=)(?!=\s)')
        s4 = (r.sub(r'\1 ', s3))
        # Add a whitespace before " if a whitespace exists after
        r = re.compile(r'(?<=\w)("|\')(?=\s)')
        s5 = (r.sub(r' \1', s4))
        # Add a whitespace after " if a whitespace exists before
        r = re.compile(r'(?<=\s)("|\')(?=\w)')
        s6 = r.sub(r'\1 ', s5)
        # Add a whitespace after " if it is the beginning of a sentence
        if s6.startswith('\"'):
            s6 = s6[1:]
            s6 = '" ' + s6
        elif s6.startswith('\''):
            s6 = s6[1:]
            s6 = '\' ' + s6
        newSentences.append(s6)
    return newSentences


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
        # Title & Author & Date
        post_entry = '//article[@class="post-entry post"]'
        meta_data = '/p[@class="post-meta"]'
        title = tree.xpath(post_entry + '/h1/text()')[0]
        author = tree.xpath(post_entry + meta_data + '/a[@rel="author"]/text()')[0]

        try:
            date = tree.xpath(post_entry + meta_data + '/span[@class="date"]/text()')[0]
        except IndexError:
            date = ''

        # Clean vertical bar from date
        date = date.replace('|', '')

        # Get all article content
        text = get_paragraphs(tree.xpath('//section[@class="post-content "]')[0].getchildren())
        text = [title] + text
        # Concat with title & sub title (separated by ".")
        # text = title + "." + author + "." + date + "." + text
        # Replace multiple whitespace with single whitespace
        # text = " ".join(text.split())

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
