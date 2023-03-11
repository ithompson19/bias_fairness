import warnings

import analyzer
from data_reader import Adult, Debug, SmallAdultEven, SmallAdultProp

def main():
    data_reader = Adult
    analyzer.all_tests(data_reader=data_reader, sensitive_attribute_column='Sex')

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main()
