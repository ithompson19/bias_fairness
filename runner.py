import warnings

import analyzer
from data_reader import Adult, Debug, Pokemon
import plotter

def main():
    analyzer.analyze_label_bias(Adult, 0.1)
    plotter.plot_all()

warnings.filterwarnings('ignore')
main()