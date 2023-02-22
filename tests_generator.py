from typing import List, Tuple
import numpy as np

def generate_tests(range_interval: float,
                       range_min: float | Tuple[str, str, float, float],
                       range_max: float | Tuple[str, str, float, float],
                       confidence_interval_test_flip_rate: Tuple[str, str, float, float] = None) -> List[Tuple[Tuple[str, str, float, float], float]]:
    if confidence_interval_test_flip_rate is None:
        tests = __label_bias_variable_args_flip_rate(range_interval, range_min, range_max)
        return list(map(lambda flip_rate: (flip_rate, 1.0), tests))
    elif isinstance(range_min, float) and isinstance(range_max, float):
        tests = __label_bias_variable_args_confidence_interval(range_interval, range_min, range_max)
        return list(map(lambda confidence_interval: (confidence_interval_test_flip_rate, confidence_interval), tests))

def __label_bias_variable_args_flip_rate(range_interval: float,
                                       range_min: Tuple[str, str, float, float],
                                       range_max: Tuple[str, str, float, float]) -> List[Tuple[str, str, float, float]]:
    if range_min[0] != range_max[0] or range_min[1] != range_max[1]:
        raise ValueError('Range min and max must have the same attribute column and value.')
    return [(range_min[0],
             range_min[1],
             flip_rate if range_max[2] > 0 else 0.0,
             flip_rate if range_max[3] > 0 else 0.0)
             for flip_rate in np.arange(
                 range_min[2] if range_min[2] > 0 else range_min[3],
                 range_max[2] + range_interval if range_max[2] > 0 else range_max[3] + range_interval,
                 range_interval).tolist()]

def __label_bias_variable_args_confidence_interval(range_interval: float, range_min: float, range_max: float) -> List[float]:
    return np.arange(range_min, range_max + range_interval, range_interval).tolist()
