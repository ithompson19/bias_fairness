import constants as const
from typing import Tuple
import pandas as pd
import numpy as np
import multiprocessing as mp
import file_handler

from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from fairlearn.metrics import demographic_parity_ratio
from functools import partial
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_reader import DataReader


def analyze_label_bias(dataReader: DataReader, 
                       range_interval: float, 
                       range_min: float = const.LABEL_BIAS_RANGE_MIN,
                       range_max: float = const.LABEL_BIAS_RANGE_MAX,
                       trial_count: int = const.TRIAL_COUNT_DEFAULT,
                       cpu_count: int = mp.cpu_count()):
    results: pd.DataFrame = pd.concat((__label_bias_trial(dataReader, range_interval, range_max, range_min, cpu_count, trial_num) for trial_num in range(1, trial_count + 1)), ignore_index=True)

    file_handler.save_results_to_file(results)

def __label_bias_trial(dataReader: DataReader, 
                       range_interval: float, 
                       range_max: float,
                       range_min: float,
                       cpu_count: int, 
                       trial_num: int = 1) -> pd.DataFrame:
    initial_data, initial_labels = dataReader.training_data()
    test_data, test_labels = dataReader.test_data()
    
    print(f'Trial {trial_num}:'.ljust(10), 'Initial', sep='', end="\r", flush=True)
    initial_estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs = const.N_JOBS)
    initial_estimator.fit(X = initial_data, y = initial_labels)
    print(' '*(12 + int((range_max - range_min) / range_interval)), '|', sep='', end="\r", flush=True)
    print(f'Trial {trial_num}:'.ljust(10), '|', sep='', end='')
    
    pool: Pool = Pool(cpu_count)
    variable_args = [(flip_rate, 1) for flip_rate in np.arange(range_min, range_max + range_interval, range_interval).tolist()]
    ret = pool.starmap(partial(__label_bias_fetch_train_constrain,
                               dataReader=dataReader,
                               initial_data=initial_data,
                               initial_labels=initial_labels, 
                               initial_estimator=initial_estimator,
                               training_sensitive_attributes=dataReader.training_sensitive_attributes(),
                               test_data=test_data,
                               test_labels=test_labels,
                               test_sensitive_attributes=dataReader.test_sensitive_attributes()),
                       variable_args)
    dataframes, failures = zip(*ret)
    trial_result: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
    trial_result.insert(0, 'Trial', trial_num)
    print(f'| Failures: {sum(failures)}', flush=True)
    return trial_result

def __label_bias_fetch_train_constrain(flip_rate: float, 
                                       confidence_threshold: float, 
                                       dataReader: DataReader, 
                                       initial_data: pd.DataFrame, 
                                       initial_labels: pd.Series, 
                                       initial_estimator: LogisticRegression, 
                                       training_sensitive_attributes: pd.DataFrame, 
                                       test_data: pd.DataFrame, 
                                       test_labels: pd.Series, 
                                       test_sensitive_attributes: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    for failures in range(const.TRIAL_MAX_ATTEMPTS):
        try:
            if flip_rate == 0:
                training_data, training_labels = initial_data, initial_labels
                estimator = initial_estimator
            else:
                training_data, training_labels = dataReader.training_data_label_bias(flip_rate, confidence_threshold, initial_estimator)
                estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs=const.N_JOBS)
                estimator.fit(X = training_data, y = training_labels)
            
            constraint = DemographicParity(difference_bound = const.DIFFERENCE_BOUND)
            mitigator_dp = ExponentiatedGradient(estimator = estimator, constraints = constraint)
            mitigator_dp.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)    
            
            prediction_unc = estimator.predict(X = test_data)
            prediction_dp = mitigator_dp.predict(X = test_data)
            
            result: pd.DataFrame = pd.DataFrame(columns=const.RESULT_COLUMNS)
            result.loc[0] = [flip_rate,
                            confidence_threshold, 
                            accuracy_score(test_labels, prediction_unc),
                            demographic_parity_ratio(y_true = test_labels, 
                                                    y_pred = prediction_unc, 
                                                    sensitive_features = test_sensitive_attributes),
                            accuracy_score(test_labels, prediction_dp),
                            demographic_parity_ratio(y_true = test_labels, 
                                                    y_pred = prediction_dp, 
                                                    sensitive_features = test_sensitive_attributes)]
        except:
            continue
        break
    print('#' if failures < const.TRIAL_MAX_ATTEMPTS - 1 else 'X', end='', flush=True)
    return result, failures