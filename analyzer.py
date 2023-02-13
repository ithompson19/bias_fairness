import constants as const
from typing import List, Tuple
import pandas as pd
import multiprocessing as mp
import file_handler
import trial_args

from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity, ExponentiatedGradient, TruePositiveRateParity
from fairlearn.metrics import demographic_parity_difference, true_positive_rate, true_negative_rate
from functools import partial
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_reader import DataReader


def analyze_label_bias(data_reader: DataReader,
                       range_min: float | Tuple[str, str, float, float],
                       range_max: float | Tuple[str, str, float, float],
                       range_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
                       confidence_interval_test_flip_rate: float | Tuple[str, str, float, float] = None,
                       trial_count: int = const.TRIAL_COUNT_DEFAULT,
                       cpu_count: int = mp.cpu_count()):
    results: pd.DataFrame = pd.concat((__label_bias_trial(data_reader=data_reader, 
                                                          range_interval=range_interval, 
                                                          range_min=range_min,
                                                          range_max=range_max, 
                                                          cpu_count=cpu_count, 
                                                          confidence_interval_test_flip_rate=confidence_interval_test_flip_rate,
                                                          trial_num=trial_num) 
                                       for trial_num in range(1, trial_count + 1)), ignore_index=True)

    file_handler.save_results(results, range_min, range_max, confidence_interval_test_flip_rate)

def __label_bias_trial(data_reader: DataReader,
                       range_interval: float,
                       range_min: float | Tuple[str, str, float, float],
                       range_max: float | Tuple[str, str, float, float],
                       cpu_count: int,
                       confidence_interval_test_flip_rate: Tuple[str, str, float, float] = None,
                       trial_num: int = 1) -> pd.DataFrame:
    initial_data, initial_labels = data_reader.training_data()
    test_data, test_labels = data_reader.test_data()
    
    if confidence_interval_test_flip_rate is None:
        variable_args = trial_args.label_bias_variable_args_flip_rate(range_interval, range_min, range_max)
        variable_args = list(map(lambda flip_rate: (flip_rate, 1.0, trial_num), variable_args))
    elif isinstance(range_min, float) and isinstance(range_max, float):
        variable_args = trial_args.label_bias_variable_args_confidence_interval(range_interval, range_min, range_max)
        variable_args = list(map(lambda confidence_interval: (confidence_interval_test_flip_rate, confidence_interval, trial_num), variable_args))
        
    print(f'Trial {trial_num}:'.ljust(10), '[ ] [', ' '*len(variable_args), ']', sep='', end="\r", flush=True)
    initial_estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs = const.N_JOBS)
    initial_estimator.fit(X = initial_data, y = initial_labels)
    print(f'Trial {trial_num}:'.ljust(10), '[I] [', sep='', end='', flush=True)
    
    pool: Pool = Pool(cpu_count)
    ret = pool.starmap(partial(__label_bias_fetch_train_constrain,
                               data_reader=data_reader,
                               initial_estimator=initial_estimator,
                               training_sensitive_attributes=data_reader.training_sensitive_attributes(),
                               test_data=test_data,
                               test_labels=test_labels,
                               test_sensitive_attributes=data_reader.test_sensitive_attributes()),
                       variable_args)
    dataframes, failures = zip(*ret)
    dataframes = filter(lambda df: not df.empty, dataframes)
    trial_result: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
    print(f'] Failures: {sum(failures)}', flush=True)
    return trial_result

def __label_bias_fetch_train_constrain(flip_rate: Tuple[str, str, float, float],
                                       confidence_threshold: float,
                                       trial_num: int,
                                       data_reader: DataReader,
                                       initial_estimator: LogisticRegression,
                                       training_sensitive_attributes: pd.DataFrame,
                                       test_data: pd.DataFrame,
                                       test_labels: pd.Series,
                                       test_sensitive_attributes: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    training_data, training_labels = data_reader.training_data_label_bias(flip_rate, confidence_threshold, initial_estimator)
    for failures in range(const.TRIAL_MAX_ATTEMPTS):
        try:
            estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs=const.N_JOBS)
            estimator.fit(X = training_data, y = training_labels)
            
            constraint_dp = DemographicParity(difference_bound = const.DIFFERENCE_BOUND)
            mitigator_dp = ExponentiatedGradient(estimator = estimator, constraints = constraint_dp)
            mitigator_dp.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
            
            constraint_eo = EqualizedOdds(difference_bound = const.DIFFERENCE_BOUND)
            mitigator_eo = ExponentiatedGradient(estimator = estimator, constraints = constraint_eo)
            mitigator_eo.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
            
            constraint_eoo = TruePositiveRateParity(difference_bound = const.DIFFERENCE_BOUND)
            mitigator_eoo = ExponentiatedGradient(estimator = estimator, constraints = constraint_eoo)
            mitigator_eoo.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
            
            constraint_erp = ErrorRateParity(difference_bound = const.DIFFERENCE_BOUND)
            mitigator_erp = ExponentiatedGradient(estimator = estimator, constraints = constraint_erp)
            mitigator_erp.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
            
            predictions = [estimator.predict(X = test_data),
                           mitigator_dp.predict(X = test_data),
                           mitigator_eo.predict(X = test_data),
                           mitigator_eoo.predict(X = test_data),
                           mitigator_erp.predict(X = test_data)]
            
            model_metrics: List[float] = []
            for prediction in predictions:
                model_metrics.append(accuracy_score(y_true = test_labels,
                                                    y_pred = prediction))
                model_metrics.append(demographic_parity_difference(y_true = test_labels,
                                                                   y_pred = prediction,
                                                                   sensitive_features = test_sensitive_attributes))
                model_metrics.append(true_positive_rate(y_true = test_labels,
                                                        y_pred = prediction))
                model_metrics.append(true_negative_rate(y_true = test_labels, 
                                                        y_pred = prediction))
    
            result: pd.DataFrame = file_handler.generate_result_row(data_reader=data_reader,
                                                                    trial_num=trial_num,
                                                                    flip_rate=flip_rate,
                                                                    confidence_threshold=confidence_threshold,
                                                                    model_metrics=model_metrics)
        except:
            continue
        break
    print('#' if failures < const.TRIAL_MAX_ATTEMPTS else 'X', end='', flush=True)
    return result if failures < const.TRIAL_MAX_ATTEMPTS else pd.DataFrame(), failures