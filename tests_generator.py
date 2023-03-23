from typing import List, Tuple
import numpy as np

from data_reader import DataReader

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

def generate_dummy_tests_for_group(data_reader: DataReader, sensitive_attribute_column: str, is_advantaged: bool):
    if is_advantaged is None:
        sensitive_attribute_value = ''
    elif is_advantaged:
        sensitive_attribute_value = data_reader.sensitive_attributes[sensitive_attribute_column]
    else:
        sensitive_attribute_value = f'-{data_reader.sensitive_attributes[sensitive_attribute_column]}'
        
    flip_uni = [((sensitive_attribute_column, sensitive_attribute_value, qual, unqual), 1.0) for qual, unqual in zip([0.2, 0.4], [0.2, 0.4])]
    flip_qual = [((sensitive_attribute_column, sensitive_attribute_value, qual, unqual), 1.0) for qual, unqual in zip([0.2, 0.4], [0.0, 0.0])]
    flip_unqual = [((sensitive_attribute_column, sensitive_attribute_value, qual, unqual), 1.0) for qual, unqual in zip([0.0, 0.0], [0.2, 0.4])]
    conf_uni = [((sensitive_attribute_column, sensitive_attribute_value, qual, unqual), conf) for qual, unqual, conf in zip([0.2, 0.4], [0.2, 0.4], [0.2, 0.4])]
    conf_qual = [((sensitive_attribute_column, sensitive_attribute_value, qual, unqual), conf) for qual, unqual, conf in zip([0.2, 0.4], [0.0, 0.0], [0.2, 0.4])]
    conf_unqual = [((sensitive_attribute_column, sensitive_attribute_value, qual, unqual), conf) for qual, unqual, conf in zip([0.0, 0.0], [0.2, 0.4], [0.2, 0.4])]
    return [[flip_uni, conf_uni],
            [flip_qual, conf_qual],
            [flip_unqual, conf_unqual]]
