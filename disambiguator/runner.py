from disambiguator.text_processor import *
from sklearn.model_selection import ParameterGrid


def intersection_scorer(w1, w2, threshold):
    intersection = list(set(w1) & set(w2))
    return int(len(intersection) > threshold)


if __name__ == '__main__':
    grid_params = dict(
        window_size=[5, 10, 15],
        inner_window_size=[1, 2, 3, 4],
        scorer=intersection_scorer,
        scorer_params=[0, 1, 2, 3, 4]
    )
    processor = TextProcessor(config['database'])
    true = pd.read_csv(config['annotation'])

    for params in ParameterGrid(grid_params):
        pred = processor.predict_simple(
            json_corpus=config['corpus'],
            window_size=params['window_size'],
            inner_window_size=params['inner_window_size'],
            scorer=params['inner_window_size'],
            scorer_params=params['scorer_params']
        )
        precision, recall = processor.precision_recall_score(pred, true, score_threshold=0)
        print(params, precision, recall)
