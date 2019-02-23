import re
import json
from glob import glob

from bs4 import BeautifulSoup


def extract_text_from_html(file):
    with open(file) as text:
        soup = BeautifulSoup(text, 'html.parser')
        [s.extract() for s in soup('nomorph')]
        [s.extract() for s in soup('title')]
        [s.append('.') for s in soup(re.compile('^h[1-6]$'))]
        text = soup.get_text().replace('\n', ' ').lstrip()
    return text


def main():
    files = glob('data/preproc/*.txt')
    corpus = [extract_text_from_html(file) for file in files]
    json.dump(corpus, open('corpus.json', 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
