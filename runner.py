import warnings

import analyzer
from data_reader import Adult, Debug, Pokemon


def main():
    analyzer.compare_label_bias(Adult, 0.1)

warnings.filterwarnings('ignore')
main()