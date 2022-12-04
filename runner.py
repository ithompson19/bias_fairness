import warnings

from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_reader import Adult, DataReader, Debug, Pokemon


def main(dataReader: DataReader):
    tr_data, tr_labels, tr_sensitive_attributes = dataReader.training_data()
    test_data, test_labels, test_sensitive_attributes = dataReader.test_data()
    
    print('Training Logistic Regression model')
    estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    estimator.fit(X = tr_data, y = tr_labels)
    
    print('Fetching label-biased data')
    lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = dataReader.training_data_label_bias(0.8, 0.99, estimator)
    lb_bl_tr_data, lb_bl_tr_labels, lb_bl_tr_sensitive_attributes = dataReader.training_data_label_bias(0.8)
    
    print('\nComparing accuracy of standard model')
    compare_accuracy(tr_data, tr_labels, tr_sensitive_attributes, test_data, test_labels, estimator)
    
    print('\nComparing accuracy of label bias')
    print('Training Logistic Regression model with label-biased data')
    lb_estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    lb_estimator.fit(X = lb_tr_data, y = lb_tr_labels)
    compare_accuracy(lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes, test_data, test_labels, lb_estimator)
    
    print('\nComparing accuracy of blind label bias')
    print('Training Logistic Regression model with blind label-biased data')
    lb_bl_estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    lb_bl_estimator.fit(X = lb_bl_tr_data, y = lb_bl_tr_labels)
    compare_accuracy(lb_bl_tr_data, lb_bl_tr_labels, lb_bl_tr_sensitive_attributes, test_data, test_labels, lb_bl_estimator)

def compare_accuracy(training_data, training_labels, training_sensitive_attributes, test_data, test_labels, estimator):
    print('Training Exponentiated Gradient mitigator from estimated model')
    constraint = DemographicParity(difference_bound = 0.01)
    mitigator = ExponentiatedGradient(estimator = estimator, constraints = constraint)
    mitigator.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
    
    prediction = estimator.predict(X = test_data)
    prediction_mitigated = mitigator.predict(X = test_data)
    
    print('Accuracy of LR: {:.5f}'.format(accuracy_score(test_labels, prediction)))
    print('Accuracy of DP: {:.5f}'.format(accuracy_score(test_labels, prediction_mitigated)))

warnings.filterwarnings('ignore')
main(Adult)