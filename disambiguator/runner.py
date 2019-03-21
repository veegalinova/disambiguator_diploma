import zipfile

import gensim
import numpy as np
from sklearn.model_selection import ParameterGrid

from disambiguator.text_processor import *


def intersection_scorer(w1, w2, threshold):
    intersection = list(set(w1) & set(w2))
    return len(intersection)


class W2VScorer:
    def __init__(self, model_file):
        with zipfile.ZipFile(model_file, 'r') as archive:
            stream = archive.open('model.bin')
            self.model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

    @staticmethod
    def concatenation(context_vectors, window_size):
        shape = context_vectors.shape[1], context_vectors.shape[0]
        vector = context_vectors.flatten()
        if vector.shape[0] < shape[0] * window_size * 2:
            vector = np.concatenate([vector, np.zeros(shape[0] * window_size * 2 - vector.shape[0])])
        return vector

    @staticmethod
    def average(context_vectors):
        vector = context_vectors.sum(axis=0) / context_vectors.shape[0]
        return vector

    @staticmethod
    def fractional_decay(context_vectors):
        pass

    @staticmethod
    def cosine_similarity(vector1, vector2):
        similarity = (vector1 * vector2).sum() / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity

    def scorer(self, context1, context2, params):
        strategy, window_size = params
        vector1, vector2 = [], []
        for word in context1:
            try:
                vector1.append(self.model.get_vector(word))
            except:
                vector1.append(np.zeros((300,)))

        for word in context2:
            try:
                vector2.append(self.model.get_vector(word))
            except:
                vector2.append(np.zeros((300,)))

        if strategy == 'average':
            vector1 = self.average(np.array(vector1))
            vector2 = self.average(np.array(vector2))
        if strategy == 'concat':
            vector1 = self.concatenation(np.array(vector1), window_size)
            vector2 = self.concatenation(np.array(vector2), window_size)

        score = self.cosine_similarity(vector1, vector2)
        return score


if __name__ == '__main__':
    scorer = W2VScorer(config['model']).scorer
    best_params, best_score, best_f = 0, 0, 0
    grid_params = dict(
        window_size=[15],
        inner_window_size=[4],
        scorer=[scorer],
        scorer_params=['average'],
        threshold=[0.4]
    )

    # 0.4

    processor = TextProcessor(config['database'])
    true = pd.read_csv(config['annotation'])
    true['text_position'] = true['text_position'].apply(str)

    for params in ParameterGrid(grid_params):
        pred = processor.predict_simple(
            json_corpus=config['corpus'],
            window_size=params['window_size'],
            context_window_size=params['inner_window_size'],
            scorer=params['scorer'],
            scorer_params=[params['scorer_params'], params['inner_window_size']]
        )
        precision, recall, f = processor.precision_recall_score(pred, true, score_threshold=params['threshold'])
        print(params, precision, recall, f)
        if precision > best_f:
            best_f = precision
            best_score = precision, recall
            best_params = params

    print(best_params, best_score)
