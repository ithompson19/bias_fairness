from typing import List, Tuple, cast
import numpy as np
import plotter as plt
import multiprocessing as mp

from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from multiprocessing import Process
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_reader import DataReader


def compare_label_bias(dataReader: DataReader, flip_interval: float):
    print(f'Threads available: {mp.cpu_count()}')
    
    print('Fetching data')
    tr_data, tr_labels, tr_sensitive_attributes = dataReader.training_data()
    test_data, test_labels, test_sensitive_attributes = dataReader.test_data()
    
    print('Training Logistic Regression model')
    estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    estimator.fit(X = tr_data, y = tr_labels)
    
    flip_rates: List[float] = np.arange(0, 1, flip_interval).tolist()
    accuracies: List[float] = []
    accuracies_mitigated: List[float] = []
    
    for flip_rate in flip_rates:
        print(f'Flip rate: {flip_rate:.2f}')
        if flip_rate == 0:
            lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = tr_data, tr_labels, tr_sensitive_attributes
            lb_estimator = estimator
        else:
            print('\tFetching flipped data')
            lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = dataReader.training_data_label_bias(flip_rate, 1, estimator)
            
            print('\tTraining Logistic Regression model with flipped data')
            lb_estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
            lb_estimator.fit(X = lb_tr_data, y = lb_tr_labels)
        
        print(f'\tTraining Exponentiated Gradient mitigator{" with flipped data" if flip_rate > 0 else ""}')
        constraint = DemographicParity(difference_bound = 0.01)
        mitigator = ExponentiatedGradient(estimator = lb_estimator, constraints = constraint)
        mitigator.fit(X = lb_tr_data, y = lb_tr_labels, sensitive_features = lb_tr_sensitive_attributes)
        
        print('\tPredicting outcomes')
        prediction = lb_estimator.predict(X = test_data)
        prediction_mitigated = mitigator.predict(X = test_data)
        
        accuracy: float = cast(float, accuracy_score(test_labels, prediction))
        accuracy_mitigated: float = cast(float, accuracy_score(test_labels, prediction_mitigated))
        print(f'\tAccuracy: {accuracy:.5f} | {accuracy_mitigated:.5f}')
        accuracies.append(accuracy)
        accuracies_mitigated.append(accuracy_mitigated)
    
    plt.plot_batch('Accuracy of Constrained and Unconstrained ML Models\nWith Label Flipping Introduced', 'Flip Rate', 'Accuracy', {
        'Unconstrained': (flip_rates, accuracies),
        'Demographic Parity': (flip_rates, accuracies_mitigated)
    })

def compare_label_bias_single(dataReader: DataReader):
    tr_data, tr_labels, tr_sensitive_attributes = dataReader.training_data()
    test_data, test_labels, test_sensitive_attributes = dataReader.test_data()
    
    print('Training Logistic Regression model')
    estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    estimator.fit(X = tr_data, y = tr_labels)
    
    print('Fetching label-biased data')
    lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes = dataReader.training_data_label_bias(0.8, 0.99, estimator)
    lb_bl_tr_data, lb_bl_tr_labels, lb_bl_tr_sensitive_attributes = dataReader.training_data_label_bias(0.8)
    
    print('\nComparing accuracy of standard model')
    __compare_accuracy(tr_data, tr_labels, tr_sensitive_attributes, test_data, test_labels, estimator)
    
    print('\nComparing accuracy of label bias')
    print('Training Logistic Regression model with label-biased data')
    lb_estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    lb_estimator.fit(X = lb_tr_data, y = lb_tr_labels)
    __compare_accuracy(lb_tr_data, lb_tr_labels, lb_tr_sensitive_attributes, test_data, test_labels, lb_estimator)
    
    print('\nComparing accuracy of blind label bias')
    print('Training Logistic Regression model with blind label-biased data')
    lb_bl_estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    lb_bl_estimator.fit(X = lb_bl_tr_data, y = lb_bl_tr_labels)
    __compare_accuracy(lb_bl_tr_data, lb_bl_tr_labels, lb_bl_tr_sensitive_attributes, test_data, test_labels, lb_bl_estimator)

def __compare_accuracy(training_data, training_labels, training_sensitive_attributes, test_data, test_labels, estimator):
    print('Training Exponentiated Gradient mitigator from estimated model')
    constraint = DemographicParity(difference_bound = 0.01)
    mitigator = ExponentiatedGradient(estimator = estimator, constraints = constraint)
    mitigator.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
    
    prediction = estimator.predict(X = test_data)
    prediction_mitigated = mitigator.predict(X = test_data)
    
    print('Accuracy of LR: {:.5f}'.format(accuracy_score(test_labels, prediction)))
    print('Accuracy of DP: {:.5f}'.format(accuracy_score(test_labels, prediction_mitigated)))