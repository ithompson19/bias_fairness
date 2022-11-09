import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def label_bias(model: LogisticRegression, 
               data: pd.DataFrame, 
               label_column_name: str, 
               sensitive_attribute_column_names: List[str], 
               threshold: float, 
               rate: float) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if threshold > 1 or threshold < 0:
        raise ValueError('Threshold must be between 0 and 1, inclusive.')
    if rate > 1 or rate < 0:
        raise ValueError('Rate must be between 0 and 1, inclusive.')
    
    confidences: np.ndarray = model.predict_proba(data)
    
    neg_label = max(confidences[:, 0])
    pos_label = max(confidences[:, 1])
    
    flippable_indexes: List[int] = []
    for i, confidence in enumerate(confidences):
        if max(confidence) < threshold:
            flippable_indexes.append(i)
    
    flipped_labels = data[label_column_name]    
    num_flipped: int = int(len(flippable_indexes) * rate)
    for i in range(num_flipped):
        flip_index: int = random.choice(flippable_indexes)
        val = neg_label if data.at(flip_index, label_column_name) == pos_label else pos_label
        flipped_labels[flip_index] = val
    data[label_column_name] = flipped_labels
        
    return data, flipped_labels, data[sensitive_attribute_column_names]

