import warnings

import analyzer
import plotter
from data_reader import Adult, DataReader, Debug, SmallAdult

def main():
    # TODO: once all changes are working, use a linter to pretty everything up
    # TODO: create CLI for program once everything is working
    dr: DataReader = SmallAdult
    min: float = 0.0
    max: float = 0.5
    conf_min: float = 0.2
    conf_max: float = 1.0
    flip_rate: float = 0.2
    flip_rate_blind(dr, min, max)
    flip_rate_blind_confidence_interval(dr, conf_min, conf_max, flip_rate)
    flip_rate_pos_neg(dr, min, max)
    flip_rate_pos_neg_confidence_interval(dr, conf_min, conf_max, flip_rate)
    flip_rate_race(dr, min, max)
    flip_rate_race_confidence_interval(dr, conf_min, conf_max, flip_rate)
    flip_rate_race_pos_neg(dr, min, max)
    flip_rate_race_pos_neg_confidence_interval(dr, conf_min, conf_max, flip_rate)
    
    
def flip_rate_blind(dataReader: DataReader, min: float, max: float):
    print('\nAnalyzing Blind Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min = min,
                                range_max = max)
    plotter.plot_all()

def flip_rate_blind_confidence_interval(dataReader: DataReader, min: float, max: float, flip_rate: float):
    print('\nAnalyzing Blind Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate=flip_rate)
    plotter.plot_all()

def flip_rate_pos_neg(dataReader: DataReader, min: float, max: float):
    print('\nAnalyzing Positive Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=(min, 0.0),
                                range_max=(max, 0.0))
    plotter.plot_all()
    print('\nAnalyzing Negative Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=(0.0, 0.0),
                                range_max=(0.0, 0.5))
    plotter.plot_all()

def flip_rate_pos_neg_confidence_interval(dataReader: DataReader, min: float, max: float, flip_rate: float):
    print('\nAnalyzing Positive Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate=(flip_rate, 0.0))
    plotter.plot_all()
    print('\nAnalyzing Negative Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate=(0.0, flip_rate))
    plotter.plot_all()
    
def flip_rate_race(dataReader: DataReader, min: float, max: float):
    print('\nAnalyzing White Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min={'Race': {'White': min}},
                                range_max={'Race': {'White': max}})
    plotter.plot_all()
    print('\nAnalyzing Non-White Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader,
                                range_min={'Race': {
                                    'Black': min, 
                                    'Asian-Pac-Islander': min, 
                                    'Amer-Indian-Eskimo': min, 
                                    'Other': min}},
                                range_max={'Race': {
                                    'Black': max, 
                                    'Asian-Pac-Islander': max, 
                                    'Amer-Indian-Eskimo': max, 
                                    'Other': max}})
    plotter.plot_all()
    
def flip_rate_race_confidence_interval(dataReader: DataReader, min: float, max: float, flip_rate: float):
    print('\nAnalyzing White Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate={'Race': {'White': flip_rate}})
    plotter.plot_all()
    print('\nAnalyzing Non-White Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate={'Race': {
                                    'Black': flip_rate, 
                                    'Asian-Pac-Islander': flip_rate, 
                                    'Amer-Indian-Eskimo': flip_rate, 
                                    'Other': flip_rate}})
    plotter.plot_all()
    
def flip_rate_race_pos_neg(dataReader: DataReader, min: float, max: float):
    print('\nAnalyzing White Positive Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min={'Race': {'White': (min, 0.0)}},
                                range_max={'Race': {'White': (max, 0.0)}})
    plotter.plot_all()
    print('\nAnalyzing White Negative Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min={'Race': {'White': (0.0, min)}},
                                range_max={'Race': {'White': (0.0, max)}})
    plotter.plot_all()
    print('\nAnalyzing Non-White Positive Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader,
                                range_min={'Race': {
                                    'Black': (min, 0.0), 
                                    'Asian-Pac-Islander': (min, 0.0), 
                                    'Amer-Indian-Eskimo': (min, 0.0), 
                                    'Other': (min, 0.0)}},
                                range_max={'Race': {
                                    'Black': (max, 0.0), 
                                    'Asian-Pac-Islander': (max, 0.0), 
                                    'Amer-Indian-Eskimo': (max, 0.0), 
                                    'Other': (max, 0.0)}})
    plotter.plot_all()
    print('\nAnalyzing Non-White Negative Flip Rate')
    analyzer.analyze_label_bias(dataReader=dataReader,
                                range_min={'Race': {
                                    'Black': (0.0, min), 
                                    'Asian-Pac-Islander': (0.0, min), 
                                    'Amer-Indian-Eskimo': (0.0, min), 
                                    'Other': (0.0, min)}},
                                range_max={'Race': {
                                    'Black': (0.0, max), 
                                    'Asian-Pac-Islander': (0.0, max), 
                                    'Amer-Indian-Eskimo': (0.0, max), 
                                    'Other': (0.0, max)}})
    plotter.plot_all()
    
def flip_rate_race_pos_neg_confidence_interval(dataReader: DataReader, min: float, max: float, flip_rate: float):
    print('\nAnalyzing White Positive Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate={'Race': {'White': (flip_rate, 0.0)}})
    plotter.plot_all()
    print('\nAnalyzing White Negative Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate={'Race': {'White': (0.0, flip_rate)}})
    plotter.plot_all()
    print('\nAnalyzing Non-White Positive Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate={'Race': {
                                    'Black': (flip_rate, 0.0), 
                                    'Asian-Pac-Islander': (flip_rate, 0.0), 
                                    'Amer-Indian-Eskimo': (flip_rate, 0.0), 
                                    'Other': (flip_rate, 0.0)}})
    plotter.plot_all()
    print('\nAnalyzing Non-White Negative Confidence Interval')
    analyzer.analyze_label_bias(dataReader=dataReader, 
                                range_min=min,
                                range_max=max,
                                confidence_interval_test_flip_rate={'Race': {
                                    'Black': (0.0, flip_rate), 
                                    'Asian-Pac-Islander': (0.0, flip_rate), 
                                    'Amer-Indian-Eskimo': (0.0, flip_rate), 
                                    'Other': (0.0, flip_rate)}})
    plotter.plot_all()

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main()