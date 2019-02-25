import re
import string
import json
from collections import namedtuple

import yaml
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from yargy.tokenizer import MorphTokenizer
from razdel import sentenize

from disambiguator.db import RutezDB

with open("config.yml", 'r') as config_file:
    config = yaml.load(config_file)

# todo: rise relation threshold to 3-4 to increase recall
# todo: unlimited text window for annotation
# todo: simplest baseline

Item = namedtuple('Item', [
    'id',
    'text',
    'span',
    'normal_form',
    'is_polysemous',
    'is_stopword'
])


class TextProcessor:
    def __init__(self, database):
        self.db = RutezDB(database)
        self.morph_tokenizer = MorphTokenizer()
        self.stop_words = stopwords.words('russian') + [s for s in string.punctuation]

    @staticmethod
    def extract_text_from_html(text):
        soup = BeautifulSoup(text, 'html.parser')
        [s.extract() for s in soup('nomorph')]
        [s.extract() for s in soup('title')]
        [s.append('.') for s in soup(re.compile('^h[1-6]$'))]
        text = soup.get_text().replace('\n', ' ').lstrip()
        return text

    def morph_tokenize(self, text):
        result = []
        text = text.lower()
        tokens = list(self.morph_tokenizer(text))
        query_items = [token.normalized for token in tokens if token.value not in self.stop_words]
        ids = self.db.select_words_db_ids(query_items)
        for token in tokens:
            id_, is_poly = ids.get(token.normalized, (None, None))
            result.append(
                Item(
                    id_,
                    token.value,
                    token.span,
                    token.normalized,
                    is_poly,
                    token.normalized is stopwords
                )
            )
        return result

    def predict_simple(self, json_corpus, window_size=None, skip_sentence_border=True, scorer=None):
        corpus = json.load(open(json_corpus, 'r'))
        result = []
        for document in corpus:
            tokens = self.morph_tokenize(document)
            document_length = len(tokens)
            sentences = list(sentenize(document))

            query_items = [str(token.id) for token in tokens if token.is_polysemous == 1]
            close_words, close_words_to_meaning, meaning_id_to_word = self.db.select_close_words(query_items)

            for index, token in enumerate(tokens):
                if token.is_polysemous == 1:

                    mono_words = []
                    token_close_words = close_words.get(token.id)

                    if not window_size:
                        window_start = 0
                        window_end = document_length

                    else:
                        window_start = max(0, index - window_size)
                        window_end = min(document_length, index + window_size)

                    for window_token in tokens[window_start: window_end]:
                        if not skip_sentence_border or window_token.text != '.':
                            if window_token.is_polysemous == 0 and window_token.id in token_close_words:
                                mono_words.append(window_token.id)
                                containing_sentence = [sentence.text
                                                       for sentence in sentences
                                                       if sentence.start <= token.span[0]
                                                       and sentence.stop >= token.span[1]
                                                       ][0]
                                meaning_id = close_words_to_meaning[token.id][window_token.id]
                                meaning = meaning_id_to_word[meaning_id]
                                result.append(dict({'sentence': containing_sentence,
                                                    'word': token.text,
                                                    'text_position': token.span,
                                                    'close_word': window_token.text,
                                                    'meaning': meaning
                                                    }))
            json.dump(result, open('result.json', 'w'), ensure_ascii=False)

    def score(self, prediction, ground_truth):
        return


if __name__ == '__main__':
    p = TextProcessor(config['database'])
    p.predict_simple(config['corpus'])
