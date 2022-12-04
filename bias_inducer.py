"""Introduces bias into a provided DataFrame. 

Methods
-------
label_bias(model: LogisticRegression, data: pd.DataFrame, label_column_name: str, rate: float, threshold: float, copy_data: bool = True) -> pd.DataFrame:
    Introduces label bias in the provided dataframe. Randomly flips labels under the confidence threshold as determined by the model, at the rate specified.
"""
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression


def label_bias(data: pd.DataFrame, 
               label_column_name: str,
               rate: float | Tuple[float, float],
               copy_data: bool = True,
               confidence_model: LogisticRegression = LogisticRegression(max_iter = 10000, n_jobs = -1), 
               confidence_threshold: float = 1.0) -> pd.DataFrame:
    """Introduces label bias in the provided dataframe. Randomly flips labels under the confidence threshold as determined by the model, at the rate specified.
    
    Parameters
    ----------
    model: LogisticRegression
        Logistic Regression model on which the confidences are based.
    data: pandas.DataFrame
        DataFrame into which bias is to be introduced.
    label_column_name: str
        Name of the column that contains the labels to be flipped.
    rate: float
        Rate at which the labels are to be flipped.
    copy_data: bool, optional
        If true, does not alter the passed-in dataset, and creates an entire new dataset. (default is True). 
    threshold: float
        Confidence threshold under which labels can be flipped.
    
    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with label bias introduced.
    
    Raises
    ------
    ValueError
        If the rate is not between 0 and 1 inclusive, the threshold is not between 0 and 1 inclusive, the label column name is not a column in the dataset, or the label column is not a column of integers with values 0 and 1.
    """
    if type(rate) == float and not 0 <= rate <= 1:  # type: ignore
        raise ValueError('Rate must be between 0 and 1, inclusive.')
    if not -1 <= confidence_threshold <= 1:
        raise ValueError('Threshold must be between 0 and 1, inclusive.')
    if not label_column_name in data.columns:
        raise ValueError('Label column must be one of the columns in the provided dataset.')
    if not (is_numeric_dtype(data[label_column_name]) and data[label_column_name].isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    
    new_data: pd.DataFrame = data.copy() if copy_data else data
    
    flippable_indexes = __get_flippable_indexes(data, confidence_model, confidence_threshold)
        
    # Flip pos and negative labels at same rate unless specified
    if type(rate) == float:
        __flip_labels(data, label_column_name, rate, flippable_indexes)  # type: ignore
    else:
        positive_flippable_indexes: List[int] = [i for i in flippable_indexes if data.at[i, label_column_name] == 1]
        negative_flippable_indexes: List[int] = [i for i in flippable_indexes if data.at[i, label_column_name] == 0]
        __flip_labels(data, label_column_name, rate[0], positive_flippable_indexes)  # type: ignore
        __flip_labels(data, label_column_name, rate[1], negative_flippable_indexes)  # type: ignore
        
    return new_data

def __get_flippable_indexes(data: pd.DataFrame,
                            confidence_model: LogisticRegression, 
                            confidence_threshold: float) -> List[int]:
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
    for i in range(int(len(flippable_indexes) * rate)):
            flip_index: int = random.choice(flippable_indexes)
            data.at[flip_index, label_column_name] = 1 - data.at[flip_index, label_column_name]
            flippable_indexes.remove(flip_index)
