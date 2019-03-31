import re
import string
import json
import logging
from tqdm import tqdm
from collections import namedtuple

import yaml
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from yargy.tokenizer import MorphTokenizer
from razdel import sentenize
from natasha import NamesExtractor

from disambiguator.db import RutezDB

logger = logging.getLogger('processor')
logger.setLevel(level=logging.INFO)

with open("config.yml", 'r') as config_file:
    config = yaml.load(config_file)

word_type_mapping = config['word_type_mapping']

Item = namedtuple('Item', [
    'id',
    'text',
    'type',
    'span',
    'normal_form',
    'is_polysemous',
    'is_stopword'
])


class TextProcessor:
    def __init__(self, database):
        self.db = RutezDB(database)
        self.morph_tokenizer = MorphTokenizer()
        self.ne_extractor = NamesExtractor()
        self.stop_words = stopwords.words('russian') + [s for s in string.punctuation]

    @staticmethod
    def extract_text_from_html(text):
        soup = BeautifulSoup(text, 'html.parser')
        [s.extract() for s in soup('nomorph')]
        [s.extract() for s in soup('title')]
        [s.append('.') for s in soup(re.compile('^h[1-6]$'))]
        text = soup.get_text().replace('\n', ' ').lstrip()
        return text

    def _find_named_entities(self, text):
        result = []
        for match in self.ne_extractor(text):
            start, stop = match.span
            result.extend(text[start: stop].lower().split(' '))
        return result

    @staticmethod
    def _find_containing_sentences(sentences, token, k):
        sentence_border = k // 2
        sentence_index = [
            index
            for index, sentence in enumerate(sentences)
            if sentence.start <= token.span[0] and sentence.stop >= token.span[1]
        ][0]
        start_sentence = max(0, sentence_index - sentence_border)
        end_sentence = min(len(sentences), sentence_index + sentence_border + 1)
        result = ' '.join(sentence.text for sentence in sentences[start_sentence: end_sentence])
        return result

    @staticmethod
    def _make_text_window(idx, size, min_idx, max_idx, tokens, type_, skip=None):
        window = []
        window_start = max(min_idx, idx - size)
        window_end = min(max_idx, idx + size + 1)
        if type_ == 'token':
            window = [t for t in tokens[window_start: window_end] if t != skip]
        elif type_ == 'text':
            window = [t.normal_form for t in tokens[window_start: window_end] if t != skip]
        elif type_ == 'form':
            window = [t.normal_form + '_' + t.type for t in tokens[window_start: window_end] if t != skip]
        return window

    def morph_tokenize(self, text):
        result = []
        named_entities = self._find_named_entities(text)
        text = text.lower()
        tokens = list(self.morph_tokenizer(text))

        tokens = [token for token in tokens if token.value not in self.stop_words and token.value not in named_entities]
        query_items = [token.normalized for token in tokens]
        ids = self.db.select_words_db_ids(query_items)
        for token in tokens:
            id_, is_poly = ids.get(token.normalized, (None, None))
            if token.value not in self.stop_words:
                if hasattr(token, 'forms'):
                    word_type = [
                        word_type_mapping.get(form, form)
                        for form in list(token.forms[0].grams.values)
                        if form.isupper()
                    ][0]

                result.append(
                    Item(
                        id_,
                        token.value,
                        word_type or None,
                        token.span,
                        token.normalized,
                        is_poly,
                        token.normalized is stopwords
                    )
                )
        return result
    
    def predict_simple(self, json_corpus, window_size=10, context_window_size=2, max_relation_order=4):
        corpus = json.load(open(json_corpus, 'r', encoding='utf-8'))
        result = []

        for document_idx, document in enumerate(tqdm(corpus)):
            tokens = self.morph_tokenize(document)
            document_length = len(tokens)
            sentences = list(sentenize(document))
            query_items = [str(token.id) for token in tokens if token.is_polysemous == 1]
            close_words, close_words_to_meaning, meaning_id_to_word = \
                self.db.select_close_words(query_items, max_relation_order)
            for token_idx, token in enumerate(tokens):
                if token.is_polysemous == 1:
                    containing_sentence = self._find_containing_sentences(sentences, token, k=3)
                    token_close_words = close_words.get(token.id)

                    search_window = self._make_text_window(
                        token_idx, window_size, 0, document_length, tokens, 'token'
                    )
                    token_context_window = self._make_text_window(
                        token_idx, context_window_size, 0, document_length, tokens, 'form', skip=token
                    )

                    for search_token_idx, search_token in enumerate(search_window):
                        if search_token.is_polysemous == 0 and search_token.id in token_close_words:

                            meaning_id = close_words_to_meaning[token.id][search_token.id]
                            meaning = meaning_id_to_word[meaning_id]
                            
                            search_context_window = self._make_text_window(
                                search_token_idx, context_window_size, 0,
                                document_length, search_window, 'form', search_token
                            )

                            result.append(
                                dict(
                                    document=document_idx,
                                    word=token.text,
                                    normalized=token.normal_form.upper(),
                                    text_position=str(list(token.span)),
                                    meaning=meaning,
                                    text=containing_sentence,
                                    search_context_window=search_context_window,
                                    token_context_window=token_context_window
                                )
                            )
        result = pd.DataFrame(result)
        return result

    @staticmethod
    def _count_score(scorer, scorer_params, data):
        result = []
        for search_window, token_window in data:
            score = scorer(search_window, token_window, scorer_params)
            result.append(score)
        return result

    def result_to_file(self, scorer, scorer_params, df_pred, score_threshold=0):
        prediction = df_pred.copy()
        prediction['score'] = pd.Series(
            self._count_score(
                scorer,
                scorer_params,
                prediction[['search_context_window', 'token_context_window']].values
            ),
            name='score'
        )
        prediction.drop(['search_context_window', 'token_context_window'], axis=1, inplace=True)
        prediction.sort_values(['document', 'text_position', 'score'], ascending=[True, True, False], inplace=True)
        prediction.drop_duplicates(subset=['document', 'text_position'], inplace=True)
        result = prediction[prediction['score'] >= score_threshold]
        result = result[['normalized', 'word', 'meaning', 'text']]
        result.to_csv(config['output'], index=False)
        return result

    def precision_recall_score(self, scorer, scorer_params, df_pred, df_true, score_threshold=0):
        prediction = df_pred.copy()
        prediction['score'] = pd.Series(
            self._count_score(
                scorer,
                scorer_params,
                prediction[['search_context_window', 'token_context_window']].values
            ),
            name='score'
        )
        prediction.drop(['search_context_window', 'token_context_window'], axis=1, inplace=True)
        prediction.sort_values(['document', 'text_position', 'score'], ascending=[False, False, True], inplace=True)
        prediction.drop_duplicates(subset=['document', 'text_position'], inplace=True)

        merged = df_true.merge(prediction, on=['document', 'text_position'])

        if score_threshold == 0:
            true_pred = merged[merged['meaning'] == merged['annotation']].shape[0]
            total_pred = merged.shape[0]
        else:
            true_pred = merged[(merged['meaning'] == merged['annotation']) &
                               (merged['score'] >= score_threshold)].shape[0]
            total_pred = merged[merged['score'] >= score_threshold].shape[0]

        if total_pred == 0:
            return 0, 0, 0

        logger.info('Total words predicted: {0}'.format(total_pred))
        precision = true_pred * 100 / total_pred
        merged.drop('meaning', axis=1, inplace=True)
        merged.drop_duplicates(inplace=True)
        total_pred = merged[merged['score'] >= score_threshold].shape[0]
        recall = total_pred * 100 / df_true.shape[0]
        f_score = 2 * precision * recall / (precision + recall)

        return precision, recall, f_score
