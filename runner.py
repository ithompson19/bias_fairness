import warnings

import analyzer
import plotter
from data_reader import Adult, Debug, Pokemon

def main():
    analyzer.analyze_label_bias(dataReader = Debug, 
                                range_interval = 0.05,
                                range_min = 0,
                                range_max = 0.5,
                                trial_count = 10)
    plotter.plot_all()

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main()