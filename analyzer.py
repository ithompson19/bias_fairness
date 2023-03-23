from matplotlib.axes import Axes
import constants as const
from typing import List, Tuple
import multiprocessing as mp
import model_trainer
import metric_analyzer
import figure_generator
import tests_generator

from data_reader import DataReader

def all_tests(data_reader: DataReader,
              sensitive_attribute_column: str,
              flip_min: float = const.LABEL_BIAS_RANGE_MIN,
              flip_max: float = const.LABEL_BIAS_RANGE_MAX,
              flip_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
              conf_flip_rate: float = const.CONFIDENCE_THRESHOLD_FLIP_RATE,
              conf_min: float = const.CONFIDENCE_THRESHOLD_RANGE_MIN,
              conf_max: float = const.CONFIDENCE_THRESHOLD_RANGE_MAX,
              conf_interval: float = const.CONFIDENCE_THRESHOLD_RANGE_INTERVAL,
              trial_count: int = const.TRIAL_COUNT_DEFAULT,
              cpu_count: int = mp.cpu_count()):
    flip_rate_tests(data_reader, sensitive_attribute_column, flip_min, flip_max, flip_interval, trial_count, cpu_count)
    confidence_threshold_tests(data_reader, sensitive_attribute_column, conf_flip_rate, conf_min, conf_max, conf_interval, trial_count, cpu_count)

def uniform_tests(data_reader: DataReader,
                  sensitive_attribute_column: str,
                  flip_min: float = const.LABEL_BIAS_RANGE_MIN,
                  flip_max: float = const.LABEL_BIAS_RANGE_MAX,
                  flip_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
                  conf_flip_rate: float = const.CONFIDENCE_THRESHOLD_FLIP_RATE,
                  conf_min: float = const.CONFIDENCE_THRESHOLD_RANGE_MIN,
                  conf_max: float = const.CONFIDENCE_THRESHOLD_RANGE_MAX,
                  conf_interval: float = const.CONFIDENCE_THRESHOLD_RANGE_INTERVAL,
                  trial_count: int = const.TRIAL_COUNT_DEFAULT,
                  cpu_count: int = mp.cpu_count(),
                  show_group_plot: bool = False):
    uniform_flip_rate_tests(data_reader, sensitive_attribute_column, flip_min, flip_max, flip_interval, trial_count, cpu_count)
    uniform_confidence_threshold_tests(data_reader, sensitive_attribute_column, conf_flip_rate, conf_min, conf_max, conf_interval, trial_count, cpu_count)
    if show_group_plot:
        figure_generator.plot_group_accuracy_fairness('Uniform', data_reader, sensitive_attribute_column)

def targeted_tests(is_advantaged: bool,
                   data_reader: DataReader,
                   sensitive_attribute_column: str,
                   flip_min: float = const.LABEL_BIAS_RANGE_MIN,
                   flip_max: float = const.LABEL_BIAS_RANGE_MAX,
                   flip_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
                   conf_flip_rate: float = const.CONFIDENCE_THRESHOLD_FLIP_RATE,
                   conf_min: float = const.CONFIDENCE_THRESHOLD_RANGE_MIN,
                   conf_max: float = const.CONFIDENCE_THRESHOLD_RANGE_MAX,
                   conf_interval: float = const.CONFIDENCE_THRESHOLD_RANGE_INTERVAL,
                   trial_count: int = const.TRIAL_COUNT_DEFAULT,
                   cpu_count: int = mp.cpu_count(),
                   show_group_plot: bool = False):
    targeted_flip_rate_tests(is_advantaged, data_reader, sensitive_attribute_column, flip_min, flip_max, flip_interval, trial_count, cpu_count)
    targeted_confidence_threshold_tests(is_advantaged, data_reader, sensitive_attribute_column, conf_flip_rate, conf_min, conf_max, conf_interval, trial_count, cpu_count)
    if show_group_plot:
        figure_generator.plot_group_accuracy_fairness('Uniform', data_reader, sensitive_attribute_column, is_advantaged)

def flip_rate_tests(data_reader: DataReader, 
                    sensitive_attribute_column: str, 
                    flip_min: float = const.LABEL_BIAS_RANGE_MIN,
                    flip_max: float = const.LABEL_BIAS_RANGE_MAX,
                    flip_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
                    trial_count: int = const.TRIAL_COUNT_DEFAULT,
                    cpu_count: int = mp.cpu_count()):
    uniform_flip_rate_tests(data_reader, sensitive_attribute_column, flip_min, flip_max, flip_interval, trial_count, cpu_count)
    targeted_flip_rate_tests(True, data_reader, sensitive_attribute_column, flip_min, flip_max, flip_interval, trial_count, cpu_count)
    targeted_flip_rate_tests(False, data_reader, sensitive_attribute_column, flip_min, flip_max, flip_interval, trial_count, cpu_count)

def confidence_threshold_tests(data_reader: DataReader,
                               sensitive_attribute_column: str,
                               flip_rate: float,
                               conf_min: float = const.CONFIDENCE_THRESHOLD_RANGE_MIN,
                               conf_max: float = const.CONFIDENCE_THRESHOLD_RANGE_MAX,
                               conf_interval: float = const.CONFIDENCE_THRESHOLD_RANGE_INTERVAL,
                               trial_count: int = const.TRIAL_COUNT_DEFAULT,
                               cpu_count: int = mp.cpu_count()):
    uniform_confidence_threshold_tests(data_reader, sensitive_attribute_column, flip_rate, conf_min, conf_max, conf_interval, trial_count, cpu_count)
    targeted_confidence_threshold_tests(True, data_reader, sensitive_attribute_column, flip_rate, conf_min, conf_max, conf_interval, trial_count, cpu_count)
    targeted_confidence_threshold_tests(False, data_reader, sensitive_attribute_column, flip_rate, conf_min, conf_max, conf_interval, trial_count, cpu_count)

def uniform_flip_rate_tests(data_reader: DataReader, 
                            sensitive_attribute_column: str, 
                            flip_min: float = const.LABEL_BIAS_RANGE_MIN,
                            flip_max: float = const.LABEL_BIAS_RANGE_MAX,
                            flip_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
                            trial_count: int = const.TRIAL_COUNT_DEFAULT,
                            cpu_count: int = mp.cpu_count()):
    tests_to_run = [(f'Uniform ({sensitive_attribute_column})', True, True), 
                    (f'Qualified ({sensitive_attribute_column})', True, False), 
                    (f'Unqualified ({sensitive_attribute_column})', False, True)]
    for message, qualified_test, unqualified_test in tests_to_run:
        __flip_rate_test(message=message, 
                         data_reader=data_reader, 
                         sensitive_attribute_column=sensitive_attribute_column, 
                         sensitive_attribute_value='',
                         qualified_test=qualified_test,
                         unqualified_test=unqualified_test,
                         flip_min=flip_min,
                         flip_max=flip_max, 
                         flip_interval=flip_interval,
                         trial_count=trial_count,
                         cpu_count=cpu_count)

def uniform_confidence_threshold_tests(data_reader: DataReader,
                                      sensitive_attribute_column: str,
                                      flip_rate: float,
                                      conf_min: float = const.CONFIDENCE_THRESHOLD_RANGE_MIN,
                                      conf_max: float = const.CONFIDENCE_THRESHOLD_RANGE_MAX,
                                      conf_interval: float = const.CONFIDENCE_THRESHOLD_RANGE_INTERVAL,
                                      trial_count: int = const.TRIAL_COUNT_DEFAULT,
                                      cpu_count: int = mp.cpu_count()):
    tests_to_run = [(f'Uniform ({sensitive_attribute_column})', True, True), 
                    (f'Qualified ({sensitive_attribute_column})', True, False), 
                    (f'Unqualified ({sensitive_attribute_column})', False, True)]
    for message, qualified_test, unqualified_test in tests_to_run:
        full_flip_rate = (sensitive_attribute_column, 
                          '', 
                          flip_rate if qualified_test else 0.0, 
                          flip_rate if unqualified_test else 0.0)
        __confidence_threshold_test(message=message,
                                    data_reader=data_reader,
                                    flip_rate=full_flip_rate,
                                    conf_min=conf_min,
                                    conf_max=conf_max,
                                    conf_interval=conf_interval,
                                    trial_count=trial_count,
                                    cpu_count=cpu_count)

def targeted_flip_rate_tests(is_advantaged: bool,
                             data_reader: DataReader,
                             sensitive_attribute_column: str,
                             flip_min: float = const.LABEL_BIAS_RANGE_MIN,
                             flip_max: float = const.LABEL_BIAS_RANGE_MAX,
                             flip_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
                             trial_count: int = const.TRIAL_COUNT_DEFAULT,
                             cpu_count: int = mp.cpu_count()):
    advantaged_val = data_reader.sensitive_attributes[sensitive_attribute_column]
    sensitive_attribute_values = data_reader.sensitive_attribute_vals(sensitive_attribute_column)
    if is_advantaged:
        base_message = advantaged_val
    elif len(sensitive_attribute_values) > 2:
        base_message = f'Non-{advantaged_val}'
    else:
        base_message = [val for val in sensitive_attribute_values if val != advantaged_val][0]
    tests_to_run = [(base_message, True, True), 
                    (f'{base_message} Qualified', True, False), 
                    (f'{base_message} Unqualified', False, True)]
    for message, qualified_test, unqualified_test in tests_to_run:
        __flip_rate_test(message=message,
                         data_reader=data_reader, 
                         sensitive_attribute_column=sensitive_attribute_column, 
                         sensitive_attribute_value=(advantaged_val if is_advantaged else f'-{advantaged_val}'),
                         qualified_test=qualified_test,
                         unqualified_test=unqualified_test,
                         flip_min=flip_min,
                         flip_max=flip_max, 
                         flip_interval=flip_interval,
                         trial_count=trial_count,
                         cpu_count=cpu_count)

def targeted_confidence_threshold_tests(is_advantaged: bool,
                                        data_reader: DataReader,
                                        sensitive_attribute_column: str,
                                        flip_rate: float,
                                        conf_min: float = const.CONFIDENCE_THRESHOLD_RANGE_MIN,
                                        conf_max: float = const.CONFIDENCE_THRESHOLD_RANGE_MAX,
                                        conf_interval: float = const.CONFIDENCE_THRESHOLD_RANGE_INTERVAL,
                                        trial_count: int = const.TRIAL_COUNT_DEFAULT,
                                        cpu_count: int = mp.cpu_count()):
    advantaged_val = data_reader.sensitive_attributes[sensitive_attribute_column]
    sensitive_attribute_values = data_reader.sensitive_attribute_vals(sensitive_attribute_column)
    if is_advantaged:
        base_message = advantaged_val
    elif len(sensitive_attribute_values) > 2:
        base_message = f'Non-{advantaged_val}'
    else:
        base_message = [val for val in sensitive_attribute_values if val != advantaged_val][0]
    tests_to_run = [(base_message, True, True), 
                    (f'{base_message} Qualified', True, False), 
                    (f'{base_message} Unqualified', False, True)]
    for message, qualified_test, unqualified_test in tests_to_run:
        full_flip_rate = (sensitive_attribute_column, 
                          f'{"" if is_advantaged else "-"}{advantaged_val}', 
                          flip_rate if qualified_test else 0.0, 
                          flip_rate if unqualified_test else 0.0)
        __confidence_threshold_test(message=message,
                                    data_reader=data_reader,
                                    flip_rate=full_flip_rate,
                                    conf_min=conf_min,
                                    conf_max=conf_max,
                                    conf_interval=conf_interval,
                                    trial_count=trial_count,
                                    cpu_count=cpu_count)

def __flip_rate_test(message: str, 
                     data_reader: DataReader, 
                     sensitive_attribute_column: str, 
                     sensitive_attribute_value: str, 
                     qualified_test: bool, 
                     unqualified_test: bool,
                     flip_min: float, 
                     flip_max: float,
                     flip_interval: float,
                     trial_count: int,
                     cpu_count: int):
    print(f'\nAnalyzing Flip Rate: {message}')
    range_min: Tuple[str, str, float, float] = (sensitive_attribute_column, sensitive_attribute_value, flip_min if qualified_test else 0.0, flip_min if unqualified_test else 0.0)
    range_max: Tuple[str, str, float, float] = (sensitive_attribute_column, sensitive_attribute_value, flip_max if qualified_test else 0.0, flip_max if unqualified_test else 0.0)
    tests = tests_generator.generate_flip_rate_tests(range_min, range_max, flip_interval)
    model_trainer.label_bias_train(data_reader, tests, trial_count, cpu_count)
    metric_analyzer.generate_metrics(data_reader, tests)
    figure_generator.plot_accuracy_fairness(data_reader, tests)

def __confidence_threshold_test(message: str,
                                data_reader: DataReader,
                                flip_rate: Tuple[str, str, float, float],
                                conf_min: float,
                                conf_max: float,
                                conf_interval: float,
                                trial_count: int,
                                cpu_count: int):
    if any(flip_rate[i] > 0 and conf_min < flip_rate[i] for i in [2, 3]):
        raise ValueError('Confidence threshold minimum must be greater than flip rate.')
    print(f'\nAnalyzing Confidence Threshold: {message}')
    tests = tests_generator.generate_confidence_interval_tests(conf_min, conf_max, conf_interval, flip_rate)
    model_trainer.label_bias_train(data_reader, tests, trial_count, cpu_count)
    metric_analyzer.generate_metrics(data_reader, tests)
    figure_generator.plot_accuracy_fairness(data_reader, tests)