import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import LabelEncoder

class DataReader:
    def __init__(self, 
                 types: Dict[str, Any], 
                 training_data_path: str, 
                 test_data_path: str, 
                 label_column_names: List[str], 
                 sensitive_attribute_column_names: List[str]) -> None:
        self.__encoder: LabelEncoder = LabelEncoder()
        
        self.__types: Dict[str, Any] = types
        self.__features: list = list(types.keys())
        self.__training_path: str = training_data_path
        self.__test_path: str = test_data_path
        self.__label_column_names: List[str] = label_column_names
        self.__sensitive_attribute_column_names: List = sensitive_attribute_column_names
        
    def __read_file(self, is_test: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Read csv data into DataFrame
        df = pd.read_csv(self.__test_path if is_test else self.__training_path, 
                           names = self.__features, 
                           dtype = self.__types, 
                           sep=r'\s*,\s*', 
                           engine='python', 
                           skiprows = 1 if is_test else 0)
        labels = df[self.__label_column_names]
        sensitive_attributes = df[self.__sensitive_attribute_column_names]
        
        return df, labels, sensitive_attributes
    
    def __encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        for colName in df.columns:
            if not pd.api.types.is_numeric_dtype(df[colName]):
                df[colName] = self.__encoder.fit_transform(y = df[colName])
        return df
    
    def __read_encoded_dataframe(self, is_test: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data, labels, sensitive_attributes = self.__read_file(is_test = is_test)
        
        data = self.__encode_dataframe(data)
        labels = self.__encode_dataframe(labels)
        sensitive_attributes = self.__encode_dataframe(sensitive_attributes)
        
        return data, labels, sensitive_attributes
    
    def training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.__read_encoded_dataframe(is_test = False)
    
    def training_data_label_bias(self, flip_rate: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data, labels, sensitive_attributes = self.__read_encoded_dataframe(is_test = False)
        
        # randomly flip labels
        
        return data, labels, sensitive_attributes
    
    def test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    label_column_names = ['Target'],
    sensitive_attribute_column_names = ['Race', 'Sex'])