import warnings

from sklearn.metrics import accuracy_score

from data_reader import Adult, DataReader, Debug, Pokemon


def test_label_bias(dataReader: DataReader, rate: float, threshold: float):
    tr_data, tr_labels, tr_sensitive_attributes = dataReader.training_data()
    lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = dataReader.training_data_label_bias(rate=rate, threshold=threshold)
    a, b, c = dataReader.training_data_label_bias(rate=rate, threshold=-1*threshold)
    
    print(f'{1-accuracy_score(tr_labels, lb_tr_labels)} | {rate * threshold}')
    # print(tr_labels, lb_tr_labels, b)

warnings.filterwarnings('ignore')
test_label_bias(Debug, 0.5, .5)