from lxml import html
import requests
import re
import os
from sys import argv, exit

path = ""
number_of_args = 3


def get_paragraphs(tree):
    """
    The method purpose is to scrap all <p> elements

    Parameter:
        tree - HTML tree (by lxml structure)
    Returns:
        String of all Paragraphs text
    """
    text = ''
    for elm in tree:
        # only go deep in divs
        if elm.tag == 'div' and \
                ('class' not in elm.attrib or elm.attrib['class'] != 'wp-caption aligncenter') \
                and elm.text_content() != '':
            text += get_paragraphs(elm)  # find paragraphs in nested nodes
        elif elm.tag == 'p' or elm.tag == 'h3':
            p = elm.text_content()
            # <p> contains only text, ignoring all scripts, images, etc.
            if p.strip() != '':
                # In case the sentence doesn't contain dot at the end of the string synthetically add dot at the end
                elementText = elm.text_content()
                if not elementText.endswith('.'):
                    elementText += '.'
                text += elementText
    return text


def split_into_sentences(text):
    """
    The method purpose is to split the text into a list of sentences
    Parameters:
        text to split
    Returns:
        sentences
    """
    # split on ".", "?", "!"
    sentences_endings = re.compile(r'[^\.!?]*[\.!?]+', re.DOTALL)
    return re.findall(sentences_endings, text)


def create_file_with_given_text(file_name, text):
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
    global path, url

    # Check for expected number of Arguments
    if len(argv) == number_of_args:
        script, url, path = argv
        print("The given path is: " + path)
    else:
        exit("Invalid number of arguments")

    try:
        # GET page
        with open(url, "r" , encoding='utf-8') as file:
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

        # Concat with title & sub title (separated by ".")
        text = title + "." + author + "." + date + "." + text
        # Replace multiple whitespace with single whitespace
        text = " ".join(text.split())

        # Step 1: save to file
        create_file_with_given_text("article.txt", text)

        # Step 2: split into sentences
        tmp_sentences = split_into_sentences(text)
        sentences = []
        for s in tmp_sentences:
            sentences.append(s.strip())

        create_file_with_given_text("article_sentences.txt", "\n".join(sentences))

        # Step 3: tokenize
        newSentences = tokenize(sentences)
        create_file_with_given_text("article_tokenized.txt", "\n".join(newSentences))

    except ConnectionError as ex:
        print("Connection to geektime failed with ({0}): {1}".format(ex.errno, ex.strerror))
        exit()
    except:
        print("General error")
        exit()


if __name__ == "__main__":
    main()
