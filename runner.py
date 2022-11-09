import warnings

from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import bias_inducer
from data_reader import Adult


def main(args):
    tr_data, tr_labels, tr_sensitive_attributes = Adult.training_data()
    test_data, test_labels, test_sensitive_attributes = Adult.test_data()
    
    estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    estimator.fit(X = tr_data, y = tr_labels)
    prediction = estimator.predict(X = test_data)
    
    constraint = DemographicParity(difference_bound = 0.01)
    
    mitigator = ExponentiatedGradient(estimator = estimator, constraints = constraint)
    mitigator.fit(X = tr_data, y = tr_labels, sensitive_features = tr_sensitive_attributes)
    prediction_mitigated = mitigator.predict(X = test_data)
    
    lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = bias_inducer.label_bias(model = estimator,
                                                                                   data = tr_data, 
                                                                                   label_column_name = Adult.label_column_name, 
                                                                                   sensitive_attribute_column_names = Adult.sensitive_attribute_column_names,
                                                                                   threshold = 0.5, 
                                                                                   rate = 0.5)
    lb_estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    lb_estimator.fit(X = lb_tr_data, y = lb_tr_labels)
    lb_prediction = lb_estimator.predict(X = test_data)
    
    lb_constraint = DemographicParity(difference_bound = 0.01)
    
    lb_mitigator = ExponentiatedGradient(estimator = lb_estimator, constraints = lb_constraint)
    lb_mitigator.fit(X = lb_tr_data, y = lb_tr_labels, sensitive_features = lb_tr_sensitive_attributes)
    lb_prediction_mitigated = lb_mitigator.predict(X = test_data)
    
    print('Accuracy of LR   : {:.5f}'.format(accuracy_score(test_labels, prediction)))
    print('Accuracy of DP   : {:.5f}'.format(accuracy_score(test_labels, prediction_mitigated)))
    print('Accuracy of LB LR: {:.5f}'.format(accuracy_score(test_labels, lb_prediction)))
    print('Accuracy of LB DP: {:.5f}'.format(accuracy_score(test_labels, lb_prediction_mitigated)))

warnings.filterwarnings('ignore')
main(None)