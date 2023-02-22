"""Allows users to automatically import datasets and introduce bias into them, through custom datasets and pre-configured ones.

Classes
-------
DataReader
    Reads data from training and test data sets and encodes them. Data in the training set can be manipulated to introduce bias before returning.

Objects
-------
Adult
    The Adult dataset (https://archive.ics.uci.edu/ml/datasets/adult)
Debug
    Simple dataset meant for debugging.
"""

from math import isclose
from pathlib import Path
from typing import Any, Dict, List, Tuple

import constants as const
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import bias_inducer


class DataReader:
    """Reads data from training and test data sets and encodes them. Data in the training set can be manipulated to introduce bias.
    
    Attributes
    ----------
    label_column_name: str
        Name of the column containing the labels in the data.
    sensitive_attribute_column_names: List[str]
        List of names of columns containing the sentitive attributes in the data.
    
    Methods
    -------
    training_data() -> Tuple[pandas.DataFrame, pandas.Series]
        Gets and encodes the training data and the label column.
    training_data_label_bias(rate: float, threshold: float, model: LogisticRegression = true) -> Tuple[pandas.DataFrame, pandas.Series]
        Gets and encodes the training data and the label column. Randomly flips values in the label column with a confidence below the threshold at the specified rate.
    test_data() -> Tuple[pandas.DataFrame, pandas.Series]
        Gets and encodes the test data and the label column.
    training_sensitive_attributes() -> pandas.DataFrame
        Gets and encodes the sensitive attribute(s) of the training data.
    test_sensitive_attributes() -> pandas.DataFrame
        Gets and encodes the sensitive attribute(s) of the test data.
    """
    
    def __init__(self,
                 types: Dict[str, Any],
                 training_data_path: str,
                 training_data_line_skip: int,
                 test_data_path: str,
                 test_data_line_skip: int,
                 label_column_name: str,
                 label_negative_positive: tuple,
                 sensitive_attributes: Dict[str, str]) -> None:
        """
        Parameters
        ----------
        types: Dict[str, Any]
            Dict object that has the names of each column in the data and the data type of the column
        training_data_path: str
            Filepath to the training data file.
        test_data_path: str
            Filepath to the test data file.
        label_column_name: str
            Names of the column to be considered as labels. Must be in the column names provided in types.
        sensitive_attribute_column_names:
            Names of each of the columns to be considered sensitive attributes. Must be a subset of the column names provided in types.
        
        Raises
        ------
        ValueError
            If the file path to the training or test data does not exist, or the column names provided in label_column_names or sensitive_attribute_column_names are not a subset of the column names provided in types.
        """
        
        self.__encoder: LabelEncoder = LabelEncoder()
        self.__types: Dict[str, Any] = types
        self.__features: list = list(types.keys())
        
        self.__training_path: Path = Path(training_data_path)
        if not (self.__training_path.is_file()):
            raise ValueError("Path to training data file does not exist.")
        self.__training_data_line_skip: int = training_data_line_skip
        
        self.__test_path: Path = Path(test_data_path)
        if not (self.__test_path.is_file()):
            raise ValueError("Path to test data file does not exist.")
        self.__test_data_line_skip: int = test_data_line_skip
        
        if label_column_name not in self.__features:
            raise ValueError("Label column name must be in the column names provided in types.")
        self.label_column_name: str = label_column_name
        
        for sensitive_attribute_column in sensitive_attributes:
            if sensitive_attribute_column not in self.__features:
                raise ValueError("Sensitive attribute column names must be a subset of the column names provided in types.")
        self.sensitive_attributes: Dict[str, str] = sensitive_attributes
        self.label_negative_positive = label_negative_positive
        
    def training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Gets and encodes the training data and the label column.
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series]
            The encoded training data and the encoded label column of the data.
        """
        
        return self.__read_encoded_dataframe(is_test = False)[:2]
    
    def training_data_label_bias(self,
                                 flip_rate: Tuple[str, str, float, float],
                                 confidence_threshold: float = 1,
                                 initial_model: LogisticRegression = LogisticRegression(max_iter=const.MAX_ITER, n_jobs=const.N_JOBS)) -> Tuple[pd.DataFrame, pd.Series]:
        if not -1 <= confidence_threshold <= 1 and not isclose(confidence_threshold, -1) and not isclose(confidence_threshold, 1):
            raise ValueError('Threshold must be between -1 and 1, inclusive.')
        if not any(0 <= rate <= 1 or isclose(rate, 0) or isclose(rate, 1) for rate in flip_rate[2:]):
            raise ValueError('Flip rates must be between 0 and 1, inclusive.')
        if not all(confidence_threshold >= rate for rate in flip_rate[2:]):
            raise ValueError('Confidence threshold must be greater than or equal to all rates.')
        
        data, labels, _ = self.__read_file(is_test = False)
        if confidence_threshold < 1 and not hasattr(initial_model, "classes_"):
            initial_model.fit(X = data, y = labels)
        
        flippable_indexes: List[int] = bias_inducer.get_flippable_indexes(data, self.label_column_name, flip_rate)
        data = self.__encode_dataframe(data)
        flippable_indexes = bias_inducer.restrict_flippable_indexes(data, flippable_indexes, initial_model, confidence_threshold)
        bias_inducer.flip_labels(data, flippable_indexes, flip_rate[2] if flip_rate[2] > 0 else flip_rate[3], self.label_column_name)
        
        labels = data[self.label_column_name]
        
        return data, labels
    
    def test_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Gets and encodes the test data and the label column.
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.DataFrame]
            The encoded test data and the encoded label column of the data.
        """
        
        return self.__read_encoded_dataframe(is_test = True)

    def training_sensitive_attributes(self) -> pd.DataFrame:
        """Gets and encodes the sensitive attribute(s) of the training data.
        
        Returns
        -------
        pandas.DataFrame
            The encoded sensitive attribue(s) of the training data.
        """
        return self.__read_encoded_dataframe(is_test=False)[2]
    
    def test_sensitive_attributes(self) -> pd.DataFrame:
        """Gets and encodes the sensitive attribute(s) of the test data.
        
        Returns
        -------
        pandas.DataFrame
            The encoded sensitive attribue(s) of the test data.
        """
        return self.__read_encoded_dataframe(is_test=True)[2]
    
    def sensitive_attribute_vals(self, attribute_name: str) -> List[str]:
        if attribute_name not in self.sensitive_attributes:
            raise ValueError(f'{attribute_name} is not a sensitive attribute.')
        priveledged: str = self.sensitive_attributes[attribute_name]
        return [priveledged, f'Non-{priveledged}']
    
    def __read_file(self, is_test: bool) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Reads the data file at the location of either the training data or test data.
        
        Parameters
        ----------
        is_test: bool
            Determines where the data should be read from. True denotes test data file and False denotes training data file.
            
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame]
            The data, the label column of the data, and the sensitive attribute(s) of the data.
        
        Raises
        ------
        IOError
            If the file being read cannot be found.
        """
        try:
            df = pd.read_csv((self.__test_path if is_test else self.__training_path), 
                            names = self.__features, 
                            dtype = self.__types, 
                            sep=r'\s*,\s*', 
                            engine='python', 
                            keep_default_na=False,
                            skiprows = self.__test_data_line_skip if is_test else self.__training_data_line_skip)
        except IOError as _:
            raise IOError(f'{"Test" if is_test else "Training"} file not found at the location specified')
        
        for sensitive_attribute_column, priveledged_value in self.sensitive_attributes.items():
            if len(df[sensitive_attribute_column].unique()) > 2:
                df.loc[df[sensitive_attribute_column] != priveledged_value, sensitive_attribute_column] = f'Non-{priveledged_value}'
        
        df[self.label_column_name] = df[self.label_column_name].map(lambda label: label.strip(' .'))
        
        if len(self.label_negative_positive) != 2:
            raise ValueError('Two values must be provided for positive/negative labels.')
        for label in self.label_negative_positive:
            if label not in list(df[self.label_column_name].unique()):
                raise ValueError('Positive / negative label not in column.')
            
        self.__encoder.fit(list(self.label_negative_positive))
        
        df[self.label_column_name] = self.__encoder.transform(y = df[self.label_column_name])
        
        return df, df[self.label_column_name], df[self.sensitive_attributes.keys()]
    
    def __encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes all non-numerical columns in the given Dataframe. 
        
        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame to be encoded.
        
        Returns
        -------
        pandas.DataFrame
            The encoded DataFrame.
        """
        
        for colName in df.columns:
            if not pd.api.types.is_numeric_dtype(df[colName]):
                df[colName] = self.__encoder.fit_transform(y = df[colName])
        return df
    
    def __read_encoded_dataframe(self, is_test: bool) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Gets the training or test data, and encodes all non-numeric columns.
        
        Parameters
        ----------
        is_test: bool
            Determines where the data should be read from. True denotes test data file and False denotes training data file.
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame]
            The encoded data, the encoded label column of the data, and the encoded sensitive attribute(s) of the data.
        """
        
        data, labels, sensitive_attributes = self.__read_file(is_test = is_test)
        
        data = self.__encode_dataframe(data)
        labels = self.__encode_dataframe(labels.to_frame()).squeeze()
        sensitive_attributes = self.__encode_dataframe(sensitive_attributes)
        
        return data, labels, sensitive_attributes
        
Adult = DataReader(*const.ADULT_PARAMS)
SmallAdult = DataReader(*const.SMALLADULT_PARAMS)
Debug = DataReader(*const.DEBUG_PARAMS)