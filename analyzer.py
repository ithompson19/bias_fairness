import constants as const
from typing import Dict, List, Tuple, cast
import pandas as pd
import numpy as np
import multiprocessing as mp
import file_handler

from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from fairlearn.metrics import demographic_parity_difference
from functools import partial
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_reader import DataReader


def analyze_label_bias(dataReader: DataReader,
                       range_interval: float = const.LABEL_BIAS_RANGE_INTERVAL, 
                       range_min: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]] = const.LABEL_BIAS_RANGE_MIN,
                       range_max: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]] = const.LABEL_BIAS_RANGE_MAX,
                       confidence_interval_test_flip_rate: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]] = None,
                       trial_count: int = const.TRIAL_COUNT_DEFAULT,
                       cpu_count: int = mp.cpu_count()):
    results: pd.DataFrame = pd.concat((__label_bias_trial(dataReader=dataReader, 
                                                          range_interval=range_interval, 
                                                          range_min=range_min,
                                                          range_max=range_max, 
                                                          cpu_count=cpu_count, 
                                                          confidence_interval_test_flip_rate=confidence_interval_test_flip_rate,
                                                          trial_num=trial_num) 
                                       for trial_num in range(1, trial_count + 1)), ignore_index=True)

    file_handler.save_results_to_file(results)

def __label_bias_trial(dataReader: DataReader, 
                       range_interval: float, 
                       range_min: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]],
                       range_max: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]],
                       cpu_count: int, 
                       confidence_interval_test_flip_rate: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]] = None,
                       trial_num: int = 1) -> pd.DataFrame:
    initial_data, initial_labels = dataReader.training_data()
    test_data, test_labels = dataReader.test_data()
    
    if confidence_interval_test_flip_rate == None:
        variable_args = __label_bias_variable_args_from_range_flip_rate(range_interval, range_min, range_max)
    elif type(range_min) == type(range_max) == float:
        variable_args = __label_bias_variable_args_from_range_confidence_interval(confidence_interval_test_flip_rate, range_interval, range_min, range_max)
    variable_args = list(map(lambda arg_set: (arg_set[0], arg_set[1], trial_num), variable_args))
        
    print(f'Trial {trial_num}:'.ljust(10), '[ ] [', ' '*len(variable_args), ']', sep='', end="\r", flush=True)
    initial_estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs = const.N_JOBS)
    initial_estimator.fit(X = initial_data, y = initial_labels)
    print(f'Trial {trial_num}:'.ljust(10), '[I] [', sep='', end='', flush=True)
    
    pool: Pool = Pool(cpu_count)
    ret = pool.starmap(partial(__label_bias_fetch_train_constrain,
                               dataReader=dataReader,
                               initial_estimator=initial_estimator,
                               training_sensitive_attributes=dataReader.training_sensitive_attributes(),
                               test_data=test_data,
                               test_labels=test_labels,
                               test_sensitive_attributes=dataReader.test_sensitive_attributes()),
                       variable_args)
    dataframes, failures = zip(*ret)
    dataframes = filter(lambda df: not df.empty, dataframes)
    trial_result: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
    print(f'] Failures: {sum(failures)}', flush=True)
    return trial_result

def __label_bias_variable_args_from_range_flip_rate(range_interval: float, 
                                   range_min: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]],
                                   range_max: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]]):
    if type(range_min) != type(range_max):
        raise ValueError(f'Provided range min and max are of different types ({type(range_min)} and {type(range_max)}).')

    if type(range_min) == float:
        range_min = cast(float, range_min)
        range_max = cast(float, range_max)
        return [(flip_rate, 1, 0) for flip_rate in np.arange(range_min, range_max + range_interval, range_interval).tolist()]
    
    elif type(range_min) == tuple:
        range_min = cast(Tuple[float, float], range_min)
        range_max = cast(Tuple[float, float], range_max)
        if not any(range_min[i] == range_max[i] == 0 for i in [0, 1]):
            raise ValueError('Provided range min and max must be only for positive->negative or negative->positive.')
        if range_max[0] > 0:
            return [((flip_rate, 0), 1, 0) for flip_rate in np.arange(range_min[0], range_max[0] + range_interval, range_interval).tolist()]
        else:
            return [((0, flip_rate), 1, 0) for flip_rate in np.arange(range_min[1], range_max[1] + range_interval, range_interval).tolist()]
    
    elif type(range_min) == dict:
        
        for attribute_name, attribute_values in range_max.items():
            for attribute_value, rate in attribute_values.items():
                if attribute_name not in range_min or attribute_value not in range_min[attribute_name]:
                    raise ValueError('Attribute names and values must be shared between range min and max.')
                if type(rate) == float:
                    return [({name: {value: flip_rate}}, 1, 0) for name in range_max.keys() for value in range_max[name].keys() for flip_rate in np.arange(range_min[attribute_name][attribute_value], range_max[attribute_name][attribute_value] + range_interval, range_interval).tolist()]
                elif rate[0] > 0:
                    return [({name: {value: (flip_rate, 0)}}, 1, 0) for name in range_max.keys() for value in range_max[name].keys() for flip_rate in np.arange(range_min[attribute_name][attribute_value][0], range_max[attribute_name][attribute_value][0] + range_interval, range_interval).tolist()]
                else:
                    return [({name: {value: (0, flip_rate)}}, 1, 0) for name in range_max.keys() for value in range_max[name].keys() for flip_rate in np.arange(range_min[attribute_name][attribute_value][1], range_max[attribute_name][attribute_value][1] + range_interval, range_interval).tolist()]

def __label_bias_variable_args_from_range_confidence_interval(flip_rate: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]], 
                                                              range_interval: float, 
                                                              range_min: float, 
                                                              range_max: float) -> List[float]:
    return [(flip_rate, confidence_interval) for confidence_interval in np.arange(range_min, range_max + range_interval, range_interval).tolist()]

def __label_bias_fetch_train_constrain(flip_rate: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]], 
                                       confidence_threshold: float, 
                                       trial_num: int,
                                       dataReader: DataReader, 
                                       initial_estimator: LogisticRegression, 
                                       training_sensitive_attributes: pd.DataFrame, 
                                       test_data: pd.DataFrame, 
                                       test_labels: pd.Series, 
                                       test_sensitive_attributes: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    training_data, training_labels = dataReader.training_data_label_bias(flip_rate, confidence_threshold, initial_estimator) 
    for failures in range(const.TRIAL_MAX_ATTEMPTS):
        try:
            estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs=const.N_JOBS)
            estimator.fit(X = training_data, y = training_labels)
            
            constraint = DemographicParity(difference_bound = const.DIFFERENCE_BOUND)
            mitigator_dp = ExponentiatedGradient(estimator = estimator, constraints = constraint)
            mitigator_dp.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
            
            prediction_unc = estimator.predict(X = test_data)
            prediction_dp = mitigator_dp.predict(X = test_data)
            
            model_metrics: List[float] = [accuracy_score(test_labels, prediction_unc),
                                  demographic_parity_difference(y_true = test_labels, 
                                                                y_pred = prediction_unc, 
                                                                sensitive_features = test_sensitive_attributes),
                                  accuracy_score(test_labels, prediction_dp),
                                  demographic_parity_difference(y_true = test_labels, 
                                                                y_pred = prediction_dp, 
                                                                sensitive_features = test_sensitive_attributes)]
    
            result: pd.DataFrame = file_handler.generate_result_row(trial_num=trial_num,
                                                                    flip_rate=dataReader.fill_incomplete_rate_object(flip_rate, confidence_threshold),
                                                                    confidence_threshold=confidence_threshold,
                                                                    model_metrics=model_metrics)
        except:
            continue
        break
    print('#' if failures < const.TRIAL_MAX_ATTEMPTS else 'X', end='', flush=True)
    return result if failures < const.TRIAL_MAX_ATTEMPTS else pd.DataFrame(), failures