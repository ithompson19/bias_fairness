import warnings

import analyzer
from data_reader import Adult, Debug, Pokemon

def main():
    analyzer.analyze_label_bias(Debug, 0.01)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()