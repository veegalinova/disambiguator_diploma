from disambiguator.text_processor import TextProcessor, config


def intersection_scorer(w1, w2, params):
    intersection = list(set(w1) & set(w2))
    return len(intersection)


if __name__ == '__main__':
    grid_params = {

    }

    processor = TextProcessor(config['database'])
    pred = processor.predict_simple(config['corpus'], config['output'], scorer=intersection_scorer)
    precision, recall = processor.precision_recall_score(pred, true, score_threshold=0)