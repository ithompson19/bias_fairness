import warnings

import analyzer
import plotter
from data_reader import Adult, Debug

def main():
    # TODO: once all changes are working, use a linter to pretty everything up
    # TODO: create CLI for program once everything is working
    analyzer.analyze_label_bias(dataReader = Debug, 
                                range_interval = 0.1,
                                range_min = 0.0,
                                range_max = 0.5,
                                trial_count = 10)
    plotter.plot_all()

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main()