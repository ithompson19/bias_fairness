"""Allows users to automatically import datasets and introduce bias into them, through custom datasets and pre-configured ones.

Classes
-------
DataReader
    Reads data from training and test data sets and encodes them. Data in the training set can be manipulated to introduce bias before returning.

Objects
-------
Adult
    The Adult dataset (https://archive.ics.uci.edu/ml/datasets/adult)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import bias_inducer


class DataReader:
    """Reads data from training and test data sets and encodes them. Data in the training set can be manipulated to introduce bias.
    
    Methods
    -------
    training_data() -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Gets and encodes the training data, the label column, and the sensitive attribute(s).
    
    test_data() -> Tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame]
        Gets and encodes the test data, the label column, and the sensitive attribute(s).
    """
    
    def __init__(self, 
                 types: Dict[str, Any], 
                 training_data_path: str, 
                 test_data_path: str, 
                 label_column_name: str, 
                 sensitive_attribute_column_names: List[str]) -> None:
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
        
        self.__test_path: Path = Path(test_data_path)
        if not (self.__test_path.is_file()):
            raise ValueError("Path to test data file does not exist.")
        
        if label_column_name not in self.__features:
            raise ValueError("Label column name must be in the column names provided in types.")
        self.label_column_name: str = label_column_name
        
        for sensitive_attribute in sensitive_attribute_column_names:
            if sensitive_attribute not in self.__features:
                raise ValueError("Sensitive attribute column names must be a subset of the column names provided in types.")
        self.sensitive_attribute_column_names: List = sensitive_attribute_column_names
        
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
            df = pd.read_csv(self.__test_path if is_test else self.__training_path, 
                            names = self.__features, 
                            dtype = self.__types, 
                            sep=r'\s*,\s*', 
                            engine='python', 
                            skiprows = 1 if is_test else 0)
        except IOError as e:
            t: str = 'Test' if is_test else 'Training'
            raise IOError('{t} file not found at the location specified')
        labels = df[self.label_column_name]
        sensitive_attributes = df[self.sensitive_attribute_column_names]
        
        return df, labels, sensitive_attributes
    
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
    
    def training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Gets and encodes the training data, the label column, and the sensitive attribute(s).
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame]
            The encoded training data, the encoded label column of the data, and the encoded sensitive attribute(s) of the data.
        """
        
        return self.__read_encoded_dataframe(is_test = False)
    
    def test_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Gets and encodes the test data, the label column, and the sensitive attribute(s).
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            The encoded test data, the encoded label column of the data, and the encoded sensitive attribute(s) of the data.
        """
        
        return self.__read_encoded_dataframe(is_test = True)
        
Adult = DataReader(
    types = {
        'Age': np.int64, 
        'Workclass': 'string',
        'fnlwgt': np.int64,
        'Education': 'string',
        'Education-Num': np.int64,
        'Marital Status': 'string',
        'Occupation': 'string',
        'Relationship': 'string',
        'Race': 'string',
        'Sex': 'string',
        'Capital Gain': np.int64,
        'Capital Loss': np.int64,
        'Hours per week': np.int64,
        'Country': 'string',
        'Target': 'string' },
    training_data_path = './Data/Adult/adult.data',
    test_data_path = './Data/Adult/adult.test',
    label_column_name = 'Target',
    sensitive_attribute_column_names = ['Race', 'Sex'])