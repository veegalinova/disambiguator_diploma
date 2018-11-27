import re
import time
import json
import string

import yaml
import pymorphy2
from razdel import sentenize, tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from yargy.tokenizer import MorphTokenizer

from src.db import RutezDB

with open("config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)


# todo : add named entities extraction?
class TextProcessor:
    def __init__(self, database=None):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words = stopwords.words('russian') + [s for s in string.punctuation]
        self.morph_tokenizer = MorphTokenizer()
        self.db = RutezDB(database)

    # todo: add sentence borders
    def morph_tokenize(self, text):
        # text = re.sub('\W+|\d+', ' ', text).lower()
        result = []
        text = text.lower()
        tokens = self.morph_tokenizer(text)
        [result.append({'orig': token.value, 'word': token.normalized, 'position': token.span})
         for token in tokens]
        return result

    @staticmethod
    def extract_text_from_htm(htm):
        # with open(file) as htm:
        soup = BeautifulSoup(htm, 'html.parser')
        [s.extract() for s in soup('nomorph')]
        [s.extract() for s in soup('title')]
        [s.append('.') for s in soup(re.compile('^h[1-6]$'))]
        text = soup.get_text().replace('\n', ' ').lstrip()
        return text

    # todo: add better name
    def return_dict_tokens(self, tokens):
        query_items = [token['word'] for token in tokens if token not in self.stop_words]
        query = self.db.select_words_db_ids(query_items)
        for token in tokens:
            res = query.get(token['word'])
            if res:
                entry_id, is_poly = res
                token['entry_id'] = entry_id
                token['is_poly'] = is_poly
        return tokens

    def find_relations(self, tokens):
        query_items = [str(token['entry_id']) for token in tokens if token.get('is_poly') == 1]
        query, meanings = self.db.select_close_words(query_items)

        for un_tok in tokens:
            if un_tok.get('is_poly') == 0:
                for token in tokens:
                    if token.get('is_poly') == 1:
                        close_words = query.get(token['entry_id'])
                        if close_words.get(un_tok['entry_id']):
                            if 'suggest' not in un_tok:
                                un_tok['suggest'] = [{token['word'] : meanings.get(close_words.get(un_tok['entry_id']))}]
                            else:
                                un_tok['suggest'].append({token['word'] : meanings.get(close_words.get(un_tok['entry_id']))})

                            if 'meaning' not in token:
                                token['meaning'] = [{meanings.get(close_words.get(un_tok['entry_id'])):un_tok['word']}]
                            else:
                                token['meaning'].append({meanings.get(close_words.get(un_tok['entry_id'])):un_tok['word']})

        return tokens


def process_file(database, file):
    processor = TextProcessor(database)
    text = processor.extract_text_from_htm(file)
    tokens = processor.morph_tokenize(text)
    tokens = processor.return_dict_tokens(tokens)
    return processor.find_relations(tokens)


if __name__ == '__main__':
    lines = json.load(open('json_text.json', 'r'))
    with open('result_file_2.txt', 'w') as f:
        for line in lines:
            res = process_file(config['database'], line)
            for word in res:
                if 'meaning' in word:
                    f.write(str.upper(word['orig']) + ' ')
                elif 'suggest' in word:
                    f.write(word['orig'] + '({})'.format(str(word['suggest'])) + ' ')
                else:
                    f.write(word['orig'] + ' ')
            f.write('\n\n')

