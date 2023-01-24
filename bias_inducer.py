"""Introduces bias into a provided DataFrame. 

Methods
-------
label_bias(model: LogisticRegression, data: pd.DataFrame, label_column_name: str, rate: float, threshold: float, copy_data: bool = True) -> pd.DataFrame:
    Introduces label bias in the provided dataframe. Randomly flips labels under the confidence threshold as determined by the model, at the rate specified.
"""
import constants as const
import random
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression


def label_bias(data: pd.DataFrame, 
               label_column_name: str,
               rate: float | Tuple[float, float] | Dict[str, Dict[str, float]] | Dict[str, Dict[str, Tuple[float, float]]], 
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
    if type(rate) == float and not 0 <= cast(float, rate) <= 1:
        raise ValueError('Rate must be between 0 and 1, inclusive.')
    if not -1 <= confidence_threshold <= 1:
        raise ValueError('Threshold must be between -1 and 1, inclusive.')
    if not label_column_name in data.columns:
        raise ValueError('Label column must be one of the columns in the provided dataset.')
    if not (is_numeric_dtype(data[label_column_name]) and data[label_column_name].isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    
    new_data: pd.DataFrame = data.copy() if copy_data else data
    flippable_indexes = __get_flippable_indexes(data, confidence_model, confidence_threshold)
    
    # flat rate, one value
    if type(rate) == float:
        rate = cast(float, rate)
        __flip_labels(data, label_column_name, rate, flippable_indexes)
    
    # pos -> neg and neg -> pos values specified
    elif type(rate) == Tuple[float, float]:
        rate = cast(Tuple[float, float], rate)
        __flip_labels(data, label_column_name, rate[0], [i for i in flippable_indexes if data.at[i, label_column_name] == 1])
        __flip_labels(data, label_column_name, rate[1], [i for i in flippable_indexes if data.at[i, label_column_name] == 0])
    
    # rates for each value of each sensitive attribute are specified
    elif type(rate) == Dict[str, Dict[str, float]]:
        rate = cast(Dict[str, Dict[str, float]], rate)
        for attribute_column_name, attribute_values in rate.items():
            for attribute_value, attribute_rate in attribute_values.items():
                __flip_labels(data, label_column_name, attribute_rate, [i for i in flippable_indexes if data.at[i, attribute_column_name] == attribute_value])
    
    # rates for each value of each sensitive attribute, separated by pos -> neg and neg -> pos, are specified
    else:
        rate = cast(Dict[str, Dict[str, Tuple[float, float]]], rate)
        for attribute_column_name, attribute_values in rate.items():
            for attribute_value, attribute_rates in attribute_values.items():
                __flip_labels(data, label_column_name, attribute_rates[0], [i for i in flippable_indexes if data.at[i, label_column_name] == 1])
                __flip_labels(data, label_column_name, attribute_rates[1], [i for i in flippable_indexes if data.at[i, label_column_name] == 0])
                
    return new_data

def __get_flippable_indexes(data: pd.DataFrame,
                            confidence_model: LogisticRegression, 
                            confidence_threshold: float) -> List[int]:
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
    flippable_indexes: List[int] = []
    
    if abs(confidence_threshold) < 1 and hasattr(confidence_model, "classes_"):
        confidences = []
        for confidence in confidence_model.predict_proba(data):
            confidences.append(max(confidence))
        confidences = np.array(confidences)
        confidence_proportion: int = int(abs(confidence_threshold) * len(confidences))
        if confidence_threshold > 0:
            flippable_indexes = confidences.argpartition(confidence_proportion - 1)[:confidence_proportion].tolist()
        else:
            flippable_indexes = confidences.argpartition(-confidence_proportion)[-confidence_proportion:].tolist()
    else:
        flippable_indexes = [*range(len(data.index))]
    
    return flippable_indexes

def __flip_labels(data: pd.DataFrame,
                  label_column_name: str,
                  rate: float,
                  flippable_indexes: List[int]):
    """Flips a portion of the labels in the data provided. 
    
    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame into which bias is to be introduced.
    label_column_name: str
        Name of the column that contains the labels to be flipped.
    rate: float
        Rate at which the labels are to be flipped.
    flippable_indexes: List[int]
        List of all of the indexes of labels that are available to flip.
    """
    for _ in range(int(len(flippable_indexes) * rate)):
            flip_index: int = random.choice(flippable_indexes)
            data.at[flip_index, label_column_name] = 1 - data.at[flip_index, label_column_name]
            flippable_indexes.remove(flip_index)
