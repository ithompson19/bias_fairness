"""Allows users to automatically import datasets and introduce bias into them, through custom datasets and pre-configured ones.

Classes
-------
DataReader
    Reads data from training and test data sets and encodes them. Data in the training set can be manipulated to introduce bias before returning.

Objects
-------
Adult
    The Adult dataset (https://archive.ics.uci.edu/ml/datasets/adult)
Pokemon
    Data from the first 6 generations of Pokemon (https://gist.github.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6)
Debug
    Simple dataset meant for debugging.
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
        self.__training_data_line_skip: int = training_data_line_skip
        
        self.__test_path: Path = Path(test_data_path)
        if not (self.__test_path.is_file()):
            raise ValueError("Path to test data file does not exist.")
        self.__test_data_line_skip: int = test_data_line_skip
        
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
            df = pd.read_csv((self.__test_path if is_test else self.__training_path), 
                            names = self.__features, 
                            dtype = self.__types, 
                            sep=r'\s*,\s*', 
                            engine='python', 
                            keep_default_na=False,
                            skiprows = self.__test_data_line_skip if is_test else self.__training_data_line_skip)
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
    
    def training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Gets and encodes the training data and the label column.
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series]
            The encoded training data and the encoded label column of the data.
        """
        
        data, labels, sensitive_attributes = self.__read_encoded_dataframe(is_test = False)
        return data, labels
    
    def training_data_label_bias(self, rate: float, threshold: float = 1, model: LogisticRegression = LogisticRegression(max_iter=10000, n_jobs=-1)) -> Tuple[pd.DataFrame, pd.Series]:
        """Gets and encodes the training data and the label column. Randomly flips values in the label column with a confidence below the threshold at the specified rate. 
        
        Parameters
        ----------
        rate: float
            The rate at which labels are flipped.
        threshold: float
            The confidence threshold under which labels may be flipped.
        model: LogisticRegression, optional
            Logistic Regression model on which the confidences are based (default is new LogisticRegression). Fitted to dataset within method if not fitted already.
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series]
            The encoded training data and the encoded label column of the data.
        
        Raises
        ------
        ValueError
            If the rate is not between 0 and 1 inclusive, or the threshold is not between -1 and 1 inclusive.
        """
        
        if not 0 <= rate <= 1:
            raise ValueError('Rate must be between 0 and 1, inclusive.')
        if not -1 <= threshold <= 1:
            raise ValueError('Threshold must be between -1 and 1, inclusive.')
        
        data, labels, sensitive_attributes = self.__read_encoded_dataframe(is_test = False)
        
        if threshold < 1 and not hasattr(model, "classes_"):
            model.fit(X = data, y = labels)
        
        data = bias_inducer.label_bias(confidence_model = model, data = data, label_column_name = self.label_column_name, rate = rate, confidence_threshold = threshold, copy_data = False)
        labels = data[self.label_column_name]
        
        return data, labels
    
    def test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Gets and encodes the test data and the label column.
        
        Returns
        -------
        Tuple[pandas.DataFrame, pandas.DataFrame]
            The encoded test data and the encoded label column of the data.
        """
        
        data, labels, sensitive_attributes = self.__read_encoded_dataframe(is_test = True)
        return data, labels

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
    training_data_line_skip = 0,
    test_data_path = './Data/Adult/adult.test',
    test_data_line_skip = 1,
    label_column_name = 'Target',
    sensitive_attribute_column_names = ['Race', 'Sex'])

Pokemon = DataReader(
    types = {
        'Number': np.int64,
        'Name': 'string',
        'Type1': 'string',
        'Type2': 'string',
        'Total': np.int64,
        'HP': np.int64,
        'Attack': np.int64,
        'Defense': np.int64,
        'Sp. Atk': np.int64,
        'Sp. Def': np.int64,
        'Speed': np.int64,
        'Generation': np.int64,
        'Legendary': 'string'
    },
    training_data_path = './Data/Pokemon/pokemon.data',
    training_data_line_skip = 0,
    test_data_path = './Data/Pokemon/pokemon.test',
    test_data_line_skip = 1,
    label_column_name = 'Legendary',
    sensitive_attribute_column_names = ['Total', 'Generation']
)

Debug = DataReader(
    types = {
        'A': np.int64,
        'B': np.int64,
        'C': np.int64
    },
    training_data_path = './Data/Debug/debug.data',
    training_data_line_skip = 0,
    test_data_path = './Data/Debug/debug.test',
    test_data_line_skip = 1,
    label_column_name = 'C',
    sensitive_attribute_column_names = ['B']
)