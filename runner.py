import warnings

import analyzer
from data_reader import DataReader, SmallAdult
import plotter

def main():
    data_reader: DataReader = SmallAdult
    
    # run_flip_rate_tests(data_reader, flip_min=0.0, flip_max=0.5)
    # run_confidence_interval_tests(data_reader, conf_min=0.2, conf_max=1.0, flip_rate=0.2)
    plotter.plot_all()
    
def flip_rate_test(message: str, data_reader: DataReader, flip_min: tuple, flip_max: float, column: str = '', val: str = '', pos_test: bool = True, neg_test: bool = True):
    print(f'\nAnalyzing Flip Rate{": " if message else ""}{message}')
    analyzer.analyze_label_bias(data_reader=data_reader,
                                range_min=(column, val, flip_min if pos_test else 0.0, flip_min if neg_test else 0.0),
                                range_max=(column, val, flip_max if pos_test else 0.0, flip_max if neg_test else 0.0))

def confidence_interval_test(message: str, data_reader: DataReader, conf_min: tuple, conf_max: float, flip_rate: tuple):
    print(f'\nAnalyzing Confidence Interval{": " if message else ""}{message}')
    analyzer.analyze_label_bias(data_reader=data_reader,
                                range_min=conf_min,
                                range_max=conf_max,
                                confidence_interval_test_flip_rate=flip_rate)
    
def run_flip_rate_tests(data_reader: DataReader, flip_min: float, flip_max: float):
    flip_rate_test('', data_reader, flip_min, flip_max)
    flip_rate_test('Positive', data_reader, flip_min, flip_max, neg_test=False)
    flip_rate_test('Negative', data_reader, flip_min, flip_max, pos_test=False)
    flip_rate_test('White', data_reader, flip_min, flip_max, column='Race', val='White')
    flip_rate_test('Non-White', data_reader, flip_min, flip_max, column='Race', val='-White')
    flip_rate_test('White Positive', data_reader, flip_min, flip_max, column='Race', val='White', neg_test=False)
    flip_rate_test('White Negative', data_reader, flip_min, flip_max, column='Race', val='White', pos_test=False)
    flip_rate_test('Non-White Positive', data_reader, flip_min, flip_max, column='Race', val='-White', neg_test=False)
    flip_rate_test('Non-White Negative', data_reader, flip_min, flip_max, column='Race', val='-White', pos_test=False)
    
def run_confidence_interval_tests(data_reader: DataReader, conf_min: float, conf_max: float, flip_rate: float):
    confidence_interval_test('', data_reader, conf_min, conf_max, ('', '', flip_rate, flip_rate))
    confidence_interval_test('Positive', data_reader, conf_min, conf_max, flip_rate=('', '', flip_rate, 0.0))
    confidence_interval_test('Negative', data_reader, conf_min, conf_max, flip_rate=('', '', 0.0, flip_rate))
    confidence_interval_test('White', data_reader, conf_min, conf_max, flip_rate=('Race', 'White', flip_rate, flip_rate))
    confidence_interval_test('Non-White', data_reader, conf_min, conf_max, flip_rate=('Race', '-White', flip_rate, flip_rate))
    confidence_interval_test('White Positive', data_reader, conf_min, conf_max, flip_rate=('Race', 'White', flip_rate, 0.0))
    confidence_interval_test('White Negative', data_reader, conf_min, conf_max, flip_rate=('Race', 'White', 0.0, flip_rate))
    confidence_interval_test('Non-White Positive', data_reader, conf_min, conf_max, flip_rate=('Race', '-White', flip_rate, 0.0))
    confidence_interval_test('Non-White Negative', data_reader, conf_min, conf_max, flip_rate=('Race', '-White', 0.0, flip_rate))

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main()
