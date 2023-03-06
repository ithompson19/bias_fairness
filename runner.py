import warnings

import analyzer
from data_reader import Adult, DataReader, Debug, SmallAdultEven, SmallAdultProp

def main():
    data_reader: DataReader = Adult
    
    run_flip_rate_tests_race(data_reader, col = 'Sex', val = 'Male', flip_min=0.0, flip_max=0.5)
    run_confidence_interval_tests_race(data_reader, col = 'Sex', val = 'Male', conf_min=0.2, conf_max=1.0, flip_rate=0.2)
    
def flip_rate_test(message: str, data_reader: DataReader, flip_min: tuple, flip_max: float, column: str, val: str = '', pos_test: bool = True, neg_test: bool = True):
    print(f'\nAnalyzing Flip Rate{": " if message else ""}{message}')
    analyzer.analyze_label_bias(data_reader=data_reader,
                                range_min=(column, val, flip_min if pos_test else 0.0, flip_min if neg_test else 0.0),
                                range_max=(column, val, flip_max if pos_test else 0.0, flip_max if neg_test else 0.0))

def confidence_interval_test(message: str, data_reader: DataReader, conf_min: tuple, conf_max: float, flip_rate: tuple):
    print(f'\nAnalyzing Confidence Threshold{": " if message else ""}{message}')
    analyzer.analyze_label_bias(data_reader=data_reader,
                                range_min=conf_min,
                                range_max=conf_max,
                                confidence_interval_test_flip_rate=flip_rate)
    
def run_flip_rate_tests_race(data_reader: DataReader, col: str, val: str, flip_min: float, flip_max: float):
    flip_rate_test('', data_reader, flip_min, flip_max, column=col)
    flip_rate_test('Positive', data_reader, flip_min, flip_max, column=col, neg_test=False)
    flip_rate_test('Negative', data_reader, flip_min, flip_max, column=col, pos_test=False)
    flip_rate_test(f'{val}', data_reader, flip_min, flip_max, column=col, val=val)
    flip_rate_test(f'Non-{val}', data_reader, flip_min, flip_max, column=col, val=f'-{val}')
    flip_rate_test(f'{val} Positive', data_reader, flip_min, flip_max, column=col, val=val, neg_test=False)
    # flip_rate_test(f'{val} Negative', data_reader, flip_min, flip_max, column=col, val=val, pos_test=False)
    # flip_rate_test(f'Non-{val} Positive', data_reader, flip_min, flip_max, column=col, val=f'-{val}', neg_test=False)
    flip_rate_test(f'Non-{val} Negative', data_reader, flip_min, flip_max, column=col, val=f'-{val}', pos_test=False)
    
def run_confidence_interval_tests_race(data_reader: DataReader, col: str, val: str, conf_min: float, conf_max: float, flip_rate: float):
    if conf_min < flip_rate:
        raise ValueError('Confidence threshold minimum must be greater than flip rate.')
    confidence_interval_test('', data_reader, conf_min, conf_max, ('', '', flip_rate, flip_rate))
    confidence_interval_test('Positive', data_reader, conf_min, conf_max, flip_rate=('', '', flip_rate, 0.0))
    confidence_interval_test('Negative', data_reader, conf_min, conf_max, flip_rate=('', '', 0.0, flip_rate))
    confidence_interval_test(f'{val}', data_reader, conf_min, conf_max, flip_rate=(col, val, flip_rate, flip_rate))
    confidence_interval_test(f'Non-{val}', data_reader, conf_min, conf_max, flip_rate=(col, f'-{val}', flip_rate, flip_rate))
    confidence_interval_test(f'{val} Positive', data_reader, conf_min, conf_max, flip_rate=(col, val, flip_rate, 0.0))
    # confidence_interval_test(f'{val} Negative', data_reader, conf_min, conf_max, flip_rate=(col, val, 0.0, flip_rate))
    # confidence_interval_test(f'Non-{val} Positive', data_reader, conf_min, conf_max, flip_rate=(col, f'-{val}', flip_rate, 0.0))
    confidence_interval_test(f'Non-{val} Negative', data_reader, conf_min, conf_max, flip_rate=(col, f'-{val}', 0.0, flip_rate))

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main()
