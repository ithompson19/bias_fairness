import constants as const
from typing import Tuple
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import model_trainer
import metric_analyzer
import figure_generator
import tests_generator

from data_reader import DataReader

# TODO: multiprocessing Pool for trials, fix printing for that

def analyze_label_bias(data_reader: DataReader,
                       range_min: float | Tuple[str, str, float, float],
                       range_max: float | Tuple[str, str, float, float],
                       range_interval: float = const.LABEL_BIAS_RANGE_INTERVAL,
                       confidence_interval_test_flip_rate: float | Tuple[str, str, float, float] = None,
                       trial_count: int = const.TRIAL_COUNT_DEFAULT,
                       cpu_count: int = mp.cpu_count()):
    tests = tests_generator.generate_tests(range_interval, range_min, range_max, confidence_interval_test_flip_rate)
    model_trainer.label_bias_train(trial_count, data_reader, tests, cpu_count)
    metric_analyzer.generate_metrics(data_reader, tests)
    figure_generator.plot_all(tests)