from lxml import html
import requests
import re
import os
from sys import argv, exit

path = ""
number_of_args = 3


def get_paragraphs(tree):
    """
    Scrap all <p> elements

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
        elif elm.tag == 'p':
            p = elm.text_content()

            # In case the sentence doesn't contain dot at the end of the string add synthetically add dot at the end
            # length = len(p)
            # if "." != p[length - 1]:
            #     p = ''.join((p, '.'))
            # <p> contains only text, ignoring all scripts, images, etc.
            if p.strip() != '':
                text += elm.text_content()
    return text


def split_into_sentences(text):
    """
    Split the text into a list of sentences
    """
    sentences_endings = re.compile(r'[^\.!?]*[\.!?]+', re.DOTALL)  # split on ".", "?", "!" (can be ".."
    return re.findall(sentences_endings, text)


def create_file_with_given_text(file_name, text):
    """
    Create new file name and Write given text to file
    """
    try:
        text_file = open(os.path.join(path, file_name), 'w', encoding='utf-8')
        text_file.write(text)
        text_file.close()
    except:
        print("Encountered an error while writing to file")


def main():
    global path, url

    # Check for expected number of Arguments
    if len(argv) == number_of_args:
        script, url, path = argv
        print("The given path is: " + path)
    else:
        exit("Invalid number of arguments")

    try:
        #  User-Agent
        headers = {'User-Agent': 'Mozilla/5.0'}
        # GET page
        page = requests.get(url, headers=headers)

        # Get HTML Tree
        tree = html.fromstring(page.content)

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

    except ConnectionError as ex:
        print("Connection to geektime failed with ({0}): {1}".format(ex.errno, ex.strerror))
        exit()
    except:
        print("General error")
        exit()


if __name__ == "__main__":
    main()
