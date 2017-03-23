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


def get_all_geek_time_posts(url, links):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    soup = BeautifulSoup(response, LXML)
    for url in soup.find_all(GUID):
        try:
            links.append(url.getText())
        except:
            print("problem in getting geek time links")
    return


def run_hw1(links, path):
    i = 1
    for link in links:
        subprocess.call([sys.executable, HW__PY, link, os.path.join(path, '')])
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
    get_all_geek_time_posts(blog_url, links)
    # write links to file
    write_file(path, "links.txt", links)
    # unit test hw1
    run_hw1(links, path)


if __name__ == "__main__":
    main()
