from cProfile import label
from sre_compile import isstring
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fairlearn.reductions import ExponentiatedGradient, DemographicParity # Exponentiated Gradient (for Adult) / GridSearch (Worse for Adult)

TYPES = {'Age': np.int64, 
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
         'Target': 'string' }
FEATURES = list(TYPES.keys())

def main(args):
    # training data and test data
    # training data -> .fit algorithms
    # test data -> .predict methods
    # labels -> 'Target'
    # sensitive_features -> 'Sex', 'Race' 
    
    # introduce bias in labels
    # get confidence of decision, and flip low confidence rows?
    
    # na_values = '?'
    dataframe = pd.read_csv('./Data/Adult/adult.test', names = FEATURES, dtype = TYPES, sep=r'\s*,\s*', engine='python', skiprows = 1)  # type: ignore
    
    encoder = LabelEncoder()
    
    labels = dataframe['Target']
    labels = encoder.fit_transform(y = labels)
    
    for typeName in TYPES:
        if isstring(TYPES[typeName]):
            dataframe[typeName] = encoder.fit_transform(y = dataframe[typeName])
    
    print(dataframe.dtypes)
    
    estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    
    estimator.fit(X = dataframe, y = labels)
    prediction = estimator.predict(X = dataframe)
    
    constraint = DemographicParity(difference_bound = 0.01)
    
    mitigator = ExponentiatedGradient(estimator = estimator, constraints = constraint)
    
    mitigator.fit(X = dataframe, y = labels, sensitive_features = dataframe['Target'])
    
    prediction_mitigated = mitigator.predict(X = dataframe)
    
    print(prediction)
    
    print(prediction_mitigated)
main(None)