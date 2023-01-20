import warnings

import analyzer
# import plotter
from data_reader import Adult, Debug, Pokemon

def main():
    analyzer.analyze_label_bias(Debug, 0.05)
    # plotter.plot_all()

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main()