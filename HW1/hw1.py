from lxml import html
import requests
import re
import os
from sys import argv, exit

path = ""
number_of_args = 3


def create_file(file_name, text):
    """
    Create new file name and Write given text to file
    """
    try:
        file = open(os.path.join(path, file_name), 'w')
        file.write(text)
        file.close()
    except:
        print("encountered an error while writing to file")


def main():
    global path, url

    # Get Arguments
    if len(argv) == number_of_args:
        script, url, path = argv
        print("The given path is: " + path)
    else:
        exit("Invalid number of arguments")

    try:
        # chrome user-agent
        headers = {
            'User-Agent': 'Mozilla/5.0'}
        # Download page
        page = requests.get(url, headers=headers)

        # Get HTML Tree
        tree = html.fromstring(page.content)

        # Title & Sub Title
        title = tree.xpath('//article/h1/text()')[0]
        print(title)

    except ConnectionError as ex:
        print("Connection to geektime failed with ({0}): {1}".format(ex.errno, ex.strerror))
        exit()
    except:
        print("General error")
        exit()


if __name__ == "__main__":
    main()
