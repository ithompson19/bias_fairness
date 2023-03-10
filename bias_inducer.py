"""Introduces bias into a provided DataFrame. 

Methods
-------
label_bias(model: LogisticRegression, data: pd.DataFrame, label_column_name: str, rate: float, threshold: float, copy_data: bool = True) -> pd.DataFrame:
    Introduces label bias in the provided dataframe. Randomly flips labels under the confidence threshold as determined by the model, at the rate specified.
"""
from math import isclose
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression

def get_flippable_indexes(data: pd.DataFrame, labels: pd.Series, flip_rate: Tuple[str, str, float, float]) -> List[int]:
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
    if not (is_numeric_dtype(labels) and labels.isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    if not bool(flip_rate[0]):
        raise ValueError('Flip rate must contain a column name.')
    if flip_rate[0] not in data.columns:
        raise ValueError(f'Column name {flip_rate[0]} not in data.')
    if flip_rate[1] and flip_rate[1].lstrip('-') not in data[flip_rate[0]].unique():
        raise ValueError(f'Value {flip_rate[1]} does not exist in column {flip_rate[0]}')
    flippable_data = data
    if flip_rate[2] != 0 and flip_rate[3] == 0:
        positive_label_indexes = labels.loc[labels == 1].index
        flippable_data = flippable_data.iloc[positive_label_indexes]
    elif flip_rate[3] != 0 and flip_rate[2] == 0:
        negative_label_indexes = labels.loc[labels == 0].index
        flippable_data = flippable_data.iloc[negative_label_indexes]
    col: str = flip_rate[0]
    privileged_val: str = flip_rate[1].lstrip('-')
    is_privileged: bool = not flip_rate[1].startswith('-')
    flippable_data: pd.DataFrame = flippable_data.loc[(flippable_data[col] == privileged_val) == is_privileged] if flip_rate[1] else flippable_data
    return list(flippable_data.index)

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

def flip_labels(labels: pd.Series,
                flip_rate: float,
                flippable_indexes: List[int]):
    if not (is_numeric_dtype(labels) and labels.isin([0,1]).all()):
        raise ValueError('Label column must be a column of integers with values 0 and 1.')
    for _ in range(int(len(flippable_indexes) * flip_rate)):
        flip_index: int = random.choice(flippable_indexes)
        labels.at[flip_index] = 1 - labels.at[flip_index]
        flippable_indexes.remove(flip_index)