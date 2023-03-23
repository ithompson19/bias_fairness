import warnings

import analyzer
from data_reader import DataReader, Adult, Debug, SmallAdultEven, SmallAdultProp

def main(data_reader: DataReader):
    analyzer.all_tests(data_reader=data_reader, sensitive_attribute_column='Sex')
    analyzer.all_tests(data_reader=data_reader, sensitive_attribute_column='Race')

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    main(Adult)
