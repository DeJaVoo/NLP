import urllib.request
from bs4 import BeautifulSoup
import os
import sys
import subprocess
import re

HW__PY = 'hw1.py'
ARTICLE = "article"
GUID = 'guid'
LXML = "lxml"


def get_all_geek_time_posts(path, root, links):
    req = urllib.request.Request(root, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    soup = BeautifulSoup(response, LXML)
    index = 1
    for url in soup.find_all(GUID):
        try:
            links.append(url.getText())
            article_req = urllib.request.Request(url.getText(), headers={'User-Agent': 'Mozilla/5.0'})
            article_response = urllib.request.urlopen(article_req)
            save_html(path, 'articles', article_response, index)
            index += 1
        except:
            print("problem in getting geek time links")
    return


def save_html(path, directory, response, index):
    if not os.path.exists(directory):
        os.makedirs(directory)

    article_path = '{path}\\{directory}\\article{index}.html'.format(path=path, directory=directory, index=index)
    with open(article_path, 'wb+') as f:
        f.write(response.read())


def run_hw1(links, path, directory):
    i = 1
    for link in links:
        article_path = '{path}\\{directory}\\article{index}.html'.format(path=path, directory=directory, index = i)
        subprocess.call([sys.executable, HW__PY, article_path, os.path.join(path, '')])
        for filename in os.listdir("."):
            if filename.startswith(ARTICLE):
                name = os.path.splitext(filename)[0]
                match = re.search(r'\d+$', name)
                if match is None:
                    ext = str(i) + '.'
                    os.rename(filename, filename.replace('.', ext))
        i += 1


def write_file(path, file_name, links):
    text_file = open(os.path.join(path, file_name), 'w')
    for link in links:
        text_file.write("%s\n" % link)
    text_file.close()


def main():
    # extract args
    script, path = sys.argv
    blog_url = "http://www.geektime.co.il/feed/"
    links = []
    # extract links
    get_all_geek_time_posts(path, blog_url, links)
    # write links to file
    write_file(path, "links.txt", links)
    # unit test hw1
    run_hw1(links, path, 'articles')


if __name__ == "__main__":
    main()
