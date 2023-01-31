"""Introduces bias into a provided DataFrame. 

Methods
-------
label_bias(model: LogisticRegression, data: pd.DataFrame, label_column_name: str, rate: float, threshold: float, copy_data: bool = True) -> pd.DataFrame:
    Introduces label bias in the provided dataframe. Randomly flips labels under the confidence threshold as determined by the model, at the rate specified.
"""
from math import isclose
import constants as const
import random
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression


def label_bias(data: pd.DataFrame, 
               label_column_name: str,
               flip_rate: Dict[str, Dict[str, Tuple[float, float]]], 
               copy_data: bool = True,
               confidence_model: LogisticRegression = LogisticRegression(max_iter = const.MAX_ITER, n_jobs = const.N_JOBS), 
               confidence_threshold: float = 1.0) -> pd.DataFrame:
    """Introduces label bias in the provided dataframe. Randomly flips labels under the confidence threshold as determined by the model, at the rate specified.
    
    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame into which bias is to be introduced.
    label_column_name: str
        Name of the column that contains the labels to be flipped.
    rate: float
        Rate at which the labels are to be flipped.
    copy_data: bool, optional
        If true, does not alter the passed-in dataset, and creates an entire new dataset. (default is True). 
    confidence_model: LogisticRegression
        Logistic Regression model on which the confidences are based.
    confidence_threshold: float
        Confidence threshold under which labels can be flipped. If the threshold is a negative value, the most confident labels are flipped instead. 
    
    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with label bias introduced.
    
    Raises
    ------
    ValueError
        If the rate is not between 0 and 1 inclusive, the threshold is not between 0 and 1 inclusive, the label column name is not a column in the dataset, or the label column is not a column of integers with values 0 and 1.
    """
    for attribute_values in flip_rate.values():
            for attribute_rates in attribute_values.values():
                if not (0 <= attribute_rates[i] <= 1 and not isclose(attribute_rates[i], 0) and not isclose(attribute_rates[i], 1) for i in [0, 1]):
                    raise ValueError('All rates must be between 0 and 1, inclusive.')
                if any(confidence_threshold < attribute_rates[i] and not isclose(confidence_threshold, attribute_rates[i]) for i in [0, 1]):
                    raise ValueError('Confidence threshold must be greater than or equal to all rates provided.')
    if not -1 <= confidence_threshold <= 1 and not isclose(confidence_threshold, -1) and not isclose(confidence_threshold, 1):
        raise ValueError('Confidence threshold must be between -1 and 1, inclusive.')
    if not label_column_name in data.columns:
        raise ValueError('Label column must be one of the columns in the provided dataset.')
    if not (is_numeric_dtype(data[label_column_name]) and data[label_column_name].isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    new_data: pd.DataFrame = data.copy() if copy_data else data
    flippable_indexes, test_rate = get_flippable_indexes(data, flip_rate, label_column_name)
    flippable_indexes = restrict_flippable_indexes(data, flippable_indexes, confidence_model, confidence_threshold)
    flip_labels(data, flippable_indexes, test_rate, label_column_name)
                    
    return new_data

def get_flippable_indexes(data: pd.DataFrame,
                            flip_rate: Dict[str, Dict[str, Tuple[float, float]]],
                            label_column_name: str) -> Tuple[List[int], float]:
    """Returns a list of all of the indexes of labels that are available to flip.
    
    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame into which bias is to be introduced.
    confidence_model: LogisticRegression
        Logistic Regression model on which the confidences are based.
    confidence_threshold: float
        Confidence threshold under which labels can be flipped. If the threshold is a negative value, the most confident labels are flipped instead. 
    
    Returns
    -------
    List[int]
        The indexes of all of the labels that are available to flip
    """
    if not label_column_name in data.columns:
        raise ValueError('Label column must be one of the columns in the provided dataset.')
    if not (is_numeric_dtype(data[label_column_name]) and data[label_column_name].isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    flippable_indexes: List[int] = []
    test_rate: float = 0
    for sensitive_attribute_column, sensitive_attribute_values in flip_rate.items():
        for sensitive_attribute_value, value_rates in sensitive_attribute_values.items():
            for i in range(len(value_rates)):
                if value_rates[i] > 0:
                    if test_rate != 0 and value_rates[i] != test_rate:
                        raise ValueError('Mutliple rates found in data')
                    flippable_indexes.extend(data.loc[(data[sensitive_attribute_column] == sensitive_attribute_value) & (data[label_column_name] == 1 - i)].index)
                    test_rate = value_rates[i]
    flippable_indexes = np.unique(np.array(flippable_indexes)).tolist()
    return flippable_indexes, test_rate

def restrict_flippable_indexes(data: pd.DataFrame,
                                    flippable_indexes: List[int],
                                    confidence_model: LogisticRegression, 
                                    confidence_threshold: float) -> List[int]:
    if not -1 <= confidence_threshold <= 1 and not isclose(confidence_threshold, -1) and not isclose(confidence_threshold, 1):
        raise ValueError('Confidence threshold must be between -1 and 1, inclusive.')
    if abs(confidence_threshold) < 1 and hasattr(confidence_model, "classes_"):
        confidences = []
        for confidence in confidence_model.predict_proba(data):
            confidences.append(max(confidence))
        confidences = np.array(confidences)
        confidences = confidences[flippable_indexes]
        
        confidence_proportion: int = int(abs(confidence_threshold) * len(confidences))
        if confidence_threshold > 0:
            flippable_indexes = confidences.argpartition(confidence_proportion - 1)[:confidence_proportion].tolist()
        else:
            flippable_indexes = confidences.argpartition(-confidence_proportion)[-confidence_proportion:].tolist()
    return flippable_indexes

def flip_labels(data: pd.DataFrame,
                  flippable_indexes: List[int],
                  flip_rate: float,
                  label_column_name: str):
    if not label_column_name in data.columns:
        raise ValueError('Label column must be one of the columns in the provided dataset.')
    if not (is_numeric_dtype(data[label_column_name]) and data[label_column_name].isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    for _ in range(int(len(flippable_indexes) * flip_rate)):
        flip_index: int = random.choice(flippable_indexes)
        data.at[flip_index, label_column_name] = 1 - data.at[flip_index, label_column_name]
        flippable_indexes.remove(flip_index)