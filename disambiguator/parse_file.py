from disambiguator.experiment_runner import W2VScorer, intersection_scorer
from disambiguator.text_processor import *

DEFAULT_DIM = 300

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('file_parse')
logger.setLevel(level=logging.INFO)


if __name__ == '__main__':
    processor = TextProcessor(config['database'])
    model = config['models'][0]
    w2v_scorer = W2VScorer(model).scorer
    prediction = processor.predict_simple(
                json_corpus=config['corpus'],
                window_size=config['window_size'][0],
                context_window_size=config['context_window_size'][0]
            )
    scorer_params = 'fractional', config['context_window_size']
    processor.result_to_file(w2v_scorer, scorer_params, prediction, config['threshold'][0])
