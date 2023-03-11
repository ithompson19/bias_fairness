from typing import List, Tuple
import numpy as np

def generate_flip_rate_tests(range_min: Tuple[str, str, float, float],
                             range_max: Tuple[str, str, float, float],
                             range_interval: float,
                             confidence_threshold: float = 1.0) -> List[Tuple[Tuple[str, str, float, float], float]]:
    if range_min[0] != range_max[0] or range_min[1] != range_max[1]:
        raise ValueError('Range min and max must have the same attribute column and value.')
    return [((range_min[0],
              range_min[1],
              flip_rate if range_max[2] > 0 else 0.0,
              flip_rate if range_max[3] > 0 else 0.0), 
             confidence_threshold)
             for flip_rate in np.arange(
                 range_min[2] if range_min[2] > 0 else range_min[3],
                 range_max[2] + range_interval if range_max[2] > 0 else range_max[3] + range_interval,
                 range_interval).tolist()]

def generate_confidence_interval_tests(range_min: float,
                                       range_max: float, 
                                       range_interval: float, 
                                       flip_rate: Tuple[str, str, float, float]) -> List[Tuple[Tuple[str, str, float, float], float]]:
    return [(flip_rate, threshold) for threshold in np.arange(range_min, range_max + range_interval, range_interval).tolist()]
