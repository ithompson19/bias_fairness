import warnings

from sklearn.metrics import accuracy_score

from data_reader import Adult, Debug, Pokemon


def test_label_bias_blind(rate: float):
    tr_data, tr_labels, tr_sensitive_attributes = Debug.training_data()
    lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = Debug.training_data_label_bias_blind(rate=rate)
    
    print(f'Percent unchanged: {accuracy_score(tr_labels, lb_tr_labels)}')
    for i, r in tr_data.iterrows():
        print(f'{r["A"]},{r["B"]}: {tr_data.at[i, "C"]} => {lb_tr_data.at[i, "C"]}')
        
def test_label_bias(rate: float, threshold: float):
    tr_data, tr_labels, tr_sensitive_attributes = Pokemon.training_data()
    lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = Pokemon.training_data_label_bias(rate=rate, threshold=threshold)
    
    print(f'Percent unchanged: {accuracy_score(tr_labels, lb_tr_labels)}')
    # for i, r in tr_data.iterrows():
    #     print(f'{r["A"]},{r["B"]}: {tr_data.at[i, "C"]} => {lb_tr_data.at[i, "C"]}')

warnings.filterwarnings('ignore')
test_label_bias(0.5, .999)