import zipfile
import logging

import gensim
import numpy as np
from sklearn.model_selection import ParameterGrid

from disambiguator.text_processor import *

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('runner')
logger.setLevel(level=logging.INFO)


def intersection_scorer(w1, w2, params):
    intersection = list(set(w1) & set(w2))
    return len(intersection) / max(len(w1), len(w2))


class W2VScorer:
    def __init__(self, model_file):
        with zipfile.ZipFile(model_file, 'r') as archive:
            stream = archive.open('model.bin')
            self.model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

    @staticmethod
    def concatenation(context_vectors, window_size):
        result = context_vectors.copy()
        vector_dim = context_vectors.shape[1]
        for _ in range(window_size * 2 - context_vectors.shape[0]):
            result = np.append(result, [np.zeros(vector_dim)], axis=0)
        vector = result.flatten()
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

    def _get_word_vectors(self, context):
        vector = []
        for word in context:
            try:
                vector.append(self.model.get_vector(word))
            except:
                pass

        if len(vector) % 2 != 0:
            vector.append(np.zeros(300))

        return np.array(vector)

    def scorer(self, context1, context2, params):
        strategy, window_size = params
        vector1 = self._get_word_vectors(context1)
        vector2 = self._get_word_vectors(context2)

        if strategy == 'average':
            vector1 = self.average(vector1)
            vector2 = self.average(vector2)

        if strategy == 'concatenation':
            vector1 = self.concatenation(vector1, window_size)
            vector2 = self.concatenation(vector2, window_size)

        if strategy == 'fractional':
            vector1 = self.fractional_decay(vector1)
            vector2 = self.fractional_decay(vector2)

        if strategy == 'inverse_fractional':
            vector1 = self.inverse_fractional_decay(vector1)
            vector2 = self.inverse_fractional_decay(vector2)

        score = self.cosine_similarity(vector1, vector2)
        return score


if __name__ == '__main__':
    scores = []
    models = config['models']
    processor = TextProcessor(config['database'])
    annotation = pd.read_csv(config['annotation'])
    annotation['text_position'] = annotation['text_position'].apply(str)

    best_params, best_score, best_f = 0, 0, 0
    grid_params = dict(
        window_size=[10],
        context_window_size=[2],
        max_relation_order=[2],
        threshold=[0.4]
    )

    for model in models:
        w2v_scorer = W2VScorer(model).scorer
        scorers = {
            'concatenation': w2v_scorer,
            'average': w2v_scorer,
            'intersection': intersection_scorer
        }

        for params in ParameterGrid(grid_params):
            prediction = processor.predict_simple(
                json_corpus=config['corpus'],
                window_size=params['window_size'],
                context_window_size=params['context_window_size']
            )
            for strategy, scorer in scorers.items():
                scorer_params = strategy, params['context_window_size']
                precision, recall, f1 = processor.precision_recall_score(
                    scorer=scorer,
                    scorer_params=scorer_params,
                    df_pred=prediction,
                    df_true=annotation,
                    score_threshold=params['threshold']
                )
                params['model'] = model
                params['strategy'] = scorer_params
                scores.append(
                    dict(
                        model=params['model'],
                        strategy=params['strategy'],
                        window_size=params['window_size'],
                        context_window_size=params['context_window_size'],
                        threshold=params['threshold'],
                        precision=precision,
                        recall=recall
                    )
                )

                logger.info(params)
                logger.info(precision)
                logger.info(recall)

                if precision > best_f:
                    best_f = precision
                    best_score = precision, recall
                    best_params = params

            scores = pd.DataFrame(scores)
            scores.to_csv(config['log_file'])

            logger.info(model)
            logger.info(best_params)
            logger.info(best_score)

        logger.info(best_params)
        logger.info(best_score)
