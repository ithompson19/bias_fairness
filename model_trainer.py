from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import List, Tuple
import pandas as pd
import constants as const
import file_handler

from fairlearn.reductions import ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from data_reader import DataReader

def label_bias_train(trial_count: int,
                     data_reader: DataReader,
                     tests: List[Tuple[Tuple[str, str, float, float], float]],
                     cpu_count: int):
    try:
        file_handler.read_models(tests)
        print('Trained models found')
        return
    except FileNotFoundError:
        pass
    file_handler.prepare_model_directory(tests)
    print('\nTraining models...')
    print('[', ' '*trial_count*len(tests), ']', sep='', end="\r", flush=True)
    print('[', sep='', end='', flush=True)
    pool: Pool = Pool(cpu_count)
    result = pool.map_async(partial(__label_bias_trial, data_reader=data_reader, tests=tests, cpu_count=cpu_count), range(1, trial_count + 1))
    result.wait()
    pool.close()
    pool.join()
    num_failures = result.get()
    print(f'] Failures: {sum(num_failures)}\n', flush=True)

def __label_bias_trial(trial_num: int,
                       data_reader: DataReader,
                       tests: List[Tuple[Tuple[str, str, float, float], float]],
                       cpu_count: int) -> int:
    
    initial_estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs = const.N_JOBS)
    initial_data, initial_labels = data_reader.training_data()
    initial_estimator.fit(X = initial_data, y = initial_labels)
    
    pool: ThreadPool = ThreadPool(cpu_count)
    async_result = pool.starmap_async(partial(__label_bias_fetch_train_constrain,
                               data_reader=data_reader,
                               initial_estimator=initial_estimator,
                               training_sensitive_attributes=data_reader.training_sensitive_attributes(tests[0][0][0])),
                       tests)
    async_result.wait()
    return_value = async_result.get()
    trained_models, num_failures = zip(*return_value)
    file_handler.save_models(tests, trial_num, trained_models)
    return sum(num_failures)

def __label_bias_fetch_train_constrain(flip_rate: Tuple[str, str, float, float],
                                       confidence_threshold: float,
                                       data_reader: DataReader,
                                       initial_estimator: LogisticRegression,
                                       training_sensitive_attributes: pd.DataFrame) -> Tuple[Tuple[list, Tuple[str, str, float, float], float], int]:
    training_data, training_labels = data_reader.training_data_label_bias(flip_rate, confidence_threshold, initial_estimator)
    for failures in range(const.TRIAL_MAX_ATTEMPTS):
        try:
            estimator = LogisticRegression(max_iter = const.MAX_ITER, n_jobs=const.N_JOBS)
            estimator.fit(X = training_data, y = training_labels)
            
            trained_models = [estimator]
            for model, _ in const.CONSTRAINED_MODELS.items():
                constraint = model(difference_bound = const.DIFFERENCE_BOUND)
                mitigator = ExponentiatedGradient(estimator = estimator, constraints = constraint)
                mitigator.fit(X = training_data,
                              y = training_labels,
                              sensitive_features = training_sensitive_attributes)
                trained_models.append(mitigator)
        except:
            continue
        break
    print('#' if failures < const.TRIAL_MAX_ATTEMPTS - 1 else 'X', end='', flush=True)
    return (trained_models if failures < const.TRIAL_MAX_ATTEMPTS else [], flip_rate, confidence_threshold), failures