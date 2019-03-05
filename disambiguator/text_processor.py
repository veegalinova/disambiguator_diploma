import re
import string
import json
from collections import namedtuple

import yaml
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from yargy.tokenizer import MorphTokenizer
from razdel import sentenize
from tqdm import tqdm

from disambiguator.db import RutezDB

with open("config.yml", 'r') as config_file:
    config = yaml.load(config_file)

# todo: rise relation threshold to 3-4 to increase recall
# todo: words in the same grammar form have same meaning?
# todo: simple grid search
# todo: vector model
# todo: words only in different sentences
# todo: remove words from text windows

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

    def predict_simple(self, json_corpus, window_size=10,
                       inner_window_size=2, scorer=None,
                       scorer_params=None, debug=False):
        corpus = json.load(open(json_corpus, 'r'))
        result = []
        for document_index, document in enumerate(tqdm(corpus)):
            tokens = self.morph_tokenize(document)
            document_length = len(tokens)
            sentences = list(sentenize(document))

            query_items = [str(token.id) for token in tokens if token.is_polysemous == 1]
            close_words, close_words_to_meaning, meaning_id_to_word = self.db.select_close_words(query_items)

            for token_index, token in enumerate(tokens):
                if token.is_polysemous == 1:
                    containing_sentence = [
                        sentence.text
                        for sentence in sentences
                        if sentence.start <= token.span[0]
                           and sentence.stop >= token.span[1]
                    ][0]
                    token_close_words = close_words.get(token.id)

                    window_start = max(0, token_index - window_size)
                    window_end = min(document_length, token_index + window_size)
                    text_window = tokens[window_start: window_end]

                    inner_window_start = max(0, token_index - inner_window_size)
                    inner_window_end = min(document_length, token_index + inner_window_size)
                    inner_window = [token.text for token in tokens[inner_window_start: inner_window_end]]

                    for window_token_index, window_token in enumerate(text_window):
                        if window_token.is_polysemous == 0 and window_token.id in token_close_words:
                            meaning_id = close_words_to_meaning[token.id][window_token.id]
                            meaning = meaning_id_to_word[meaning_id]

                            token_window_start = max(
                                0, window_start + window_token_index - inner_window_size
                            )
                            token_window_end = min(
                                document_length, window_start + window_token_index + inner_window_size
                            )
                            token_window = [token.text for token in tokens[token_window_start: token_window_end]]

                            if scorer:
                                score = scorer(token_window, inner_window, scorer_params)
                            else:
                                score = 0

                            if debug:
                                meanings = self.db.select_poly_entries_meanings(ids=[str(token.id)])
                                result.append(dict(
                                    document=document_index, sentence=containing_sentence, word_index=token.id,
                                    word=token.text, text_position=str(token.span), close_word_index=window_token.id,
                                    close_word=window_token.text, meaning_id=meaning_id,
                                    meaning=meaning, meanings=meanings, baseline=score
                                ))

                            else:
                                result.append(dict(
                                    document=document_index,
                                    text_position=str(token.span),
                                    meaning=meaning,
                                    baseline=score
                                ))

            result = pd.DataFrame(result)
            result.drop_duplicates(inplace=True)
            return result

    @staticmethod
    def precision_recall_score(df_pred, df_true, score_threshold=0):
        merged = df_true.merge(df_pred, on=['document', 'text_position'])

        true_pred = merged[(merged['meaning'] == merged['annotation']) &
                           (merged['baseline'] > score_threshold)].shape[0]
        total_pred = merged[merged['baseline'] > 0].shape[0]
        precision = true_pred * 100 / total_pred

        merged.drop('meaning', axis=1, inplace=True)
        merged.drop_duplicates(inplace=True)
        recall = merged.shape[0] * 100 / df_true.shape[0]
        return precision, recall
