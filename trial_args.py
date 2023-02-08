from typing import List, Tuple
import numpy as np

def label_bias_variable_args_flip_rate(range_interval: float, range_min: Tuple[str, str, float, float], range_max: Tuple[str, str, float, float]):
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

def label_bias_variable_args_confidence_interval(range_interval: float, range_min: float, range_max: float) -> List[float]:
    return np.arange(range_min, range_max + range_interval, range_interval).tolist()
