"""Introduces bias into a provided DataFrame. 

Methods
-------
label_bias(model: LogisticRegression, data: pd.DataFrame, label_column_name: str, rate: float, threshold: float, copy_data: bool = True) -> pd.DataFrame:
    Introduces label bias in the provided dataframe. Randomly flips labels under the confidence threshold as determined by the model, at the rate specified.
"""
import random
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression


def label_bias(model: LogisticRegression, 
               data: pd.DataFrame, 
               label_column_name: str,
               rate: float,
               threshold: float,
               copy_data: bool = True) -> pd.DataFrame:
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
    threshold: float
        Confidence threshold under which labels can be flipped.
    copy_data: bool, optional
        If true, does not alter the passed-in dataset, and creates an entire new dataset. (default is True). 
    
    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with label bias introduced.
    
    Raises
    ------
    ValueError
        If the rate is not between 0 and 1 inclusive, the threshold is not between 0.5 and 1 inclusive, the label column name is not a column in the dataset, or the label column is not a column of integers with values 0 and 1.
    """
    if rate < 0 or rate > 1:
        raise ValueError('Rate must be between 0 and 1, inclusive.')
    if threshold < 0.5 or threshold > 1:
        raise ValueError('Threshold must be between 0.5 and 1, inclusive.')
    if not label_column_name in data.columns:
        raise ValueError('Label column must be one of the columns in the provided dataset.')
    if not (is_numeric_dtype(data[label_column_name]) and data[label_column_name].isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    
    new_data: pd.DataFrame = data.copy() if copy_data else data
    confidences: np.ndarray = model.predict_proba(data)
    
    flippable_indexes: List[int] = []
    for i, confidence in enumerate(confidences):
        if max(confidence) < threshold:
            flippable_indexes.append(i)
    num_flippable: int = int(len(flippable_indexes) * rate)
    
    for i in range(num_flippable):
        flip_index: int = random.choice(flippable_indexes)
        data.at[flip_index, label_column_name] = 1 - data.at[flip_index, label_column_name]
        flippable_indexes.remove(flip_index)
        
    return new_data

def label_bias_blind(data: pd.DataFrame,
                     label_column_name: str,
                     rate: float,
                     copy_data: bool = True) -> pd.DataFrame:
    """Introduces label bias in the provided dataframe. Randomly flips labels at the rate specified.
    
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
    
    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with label bias introduced.
    
    Raises
    ------
    ValueError
        If the rate is not between 0 and 1 inclusive, the label column name is not a column in the dataset, or the label column is not a column of integers with values 0 and 1.
    """
    if rate < 0 or rate > 1:
        raise ValueError('Rate must be between 0 and 1, inclusive.')
    if not label_column_name in data.columns:
        raise ValueError('Label column must be one of the columns in the provided dataset.')
    if not (is_numeric_dtype(data[label_column_name]) and data[label_column_name].isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    
    new_data: pd.DataFrame = data.copy() if copy_data else data
    
    num_flippable: int = int(len(data[label_column_name]) * rate)
    flippable_indexes: List[int] = [*range(num_flippable)]
    
    for i in range(num_flippable):
        flip_index: int = random.choice(flippable_indexes)
        new_data.at[flip_index, label_column_name] = 1 - new_data.at[flip_index, label_column_name]
        flippable_indexes.remove(flip_index)
    
    return new_data