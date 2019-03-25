import zipfile
import logging

import gensim
import numpy as np
from sklearn.model_selection import ParameterGrid

from disambiguator.text_processor import *

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('runner')
logger.setLevel(level=logging.INFO)


def intersection_scorer(w1, w2):
    intersection = list(set(w1) & set(w2))
    return len(intersection) / max(len(w1), len(w2))


class W2VScorer:
    def __init__(self, model_file):
        with zipfile.ZipFile(model_file, 'r') as archive:
            stream = archive.open('model.bin')
            self.model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

    @staticmethod
    def concatenation(context_vectors):
        vector = context_vectors.flatten()
        return vector

    @staticmethod
    def average(context_vectors):
        vector = context_vectors.sum(axis=0) / context_vectors.shape[0]
        return vector

    @staticmethod
    def fractional_decay(context_vectors):
        window_size = context_vectors.shape[0] // 2
        multiplier_matrix = np.array([
            (window_size + 1 - abs(i))/window_size
            for i in range(-1 * window_size, window_size + 1) if i != 0],
            dtype=np.float32
        )
        vector = (context_vectors.T * multiplier_matrix).T.sum(axis=0)
        return vector

    @staticmethod
    def inverse_fractional_decay(context_vectors):
        window_size = context_vectors.shape[0] // 2
        multiplier_matrix = np.array([
            1 / abs(window_size + 1 - abs(i)) for i in range(-window_size, window_size + 1) if i != 0],
            dtype=np.float32
        )
        vector = (context_vectors.T * multiplier_matrix).T.sum(axis=0)
        return vector

    @staticmethod
    def cosine_similarity(vector1, vector2):
        similarity = (vector1 * vector2).sum() / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity

    def _get_word_vectors(self, context, window_size):
        vector = []
        for word in context:
            try:
                vector.append(self.model.get_vector(word))
            except:
                vector.append(np.zeros((300,)))

        if len(vector) < window_size * 2:
            for i in range(window_size * 2 - len(vector)):
                vector.append(np.zeros(shape=(300,)))
        return np.array(vector)

    def scorer(self, context1, context2, params):
        strategy, window_size = params
        print(strategy)
        vector1 = self._get_word_vectors(context1, window_size)
        vector2 = self._get_word_vectors(context2, window_size)

        if strategy == 'average':
            vector1 = self.average(vector1)
            vector2 = self.average(vector2)

        if strategy == 'concat':
            vector1 = self.concatenation(vector1)
            vector2 = self.concatenation(vector2)

        if strategy == 'fractional':
            vector1 = self.fractional_decay(vector1)
            vector2 = self.fractional_decay(vector2)

        if strategy == 'inverse_fractional':
            vector1 = self.inverse_fractional_decay(vector1)
            vector2 = self.inverse_fractional_decay(vector2)

        score = self.cosine_similarity(vector1, vector2)
        return score


if __name__ == '__main__':
    scorer = W2VScorer(config['model']).scorer
    best_params, best_score, best_f = 0, 0, 0
    grid_params = dict(
        window_size=[15],
        context_window_size=[4],
        max_relation_order=[2],
        scorer=[scorer],
        scorer_params=['fractional', 'inverse_fractional', 'average', 'concatenation'],
        threshold=[0.4]
    )

    processor = TextProcessor(config['database'])
    true = pd.read_csv(config['annotation'])
    true['text_position'] = true['text_position'].apply(str)

    scores = []

    for params in ParameterGrid(grid_params):
        pred = processor.predict_simple(
            json_corpus=config['corpus'],
            scorer=params['scorer'],
            window_size=params['window_size'],
            context_window_size=params['context_window_size'],
            scorer_params=[params['scorer_params'], params['context_window_size']]
        )
        precision, recall, f1 = processor.precision_recall_score(pred, true)
        params['scorer'] = params['scorer'].__name__
        scores.append(dict(params=params, precision=precision, recall=recall))

        logger.info(params)
        logger.info(precision)
        logger.info(recall)

        if precision > best_f:
            best_f = precision
            best_score = precision, recall
            best_params = params

    scores = pd.DataFrame(scores)
    scores.to_csv('experiment1.csv')

    logger.info(best_params)
    logger.info(best_score)
