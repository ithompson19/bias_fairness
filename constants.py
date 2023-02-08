"""Holds all constants.
"""
import numpy as np

# ML Models
MAX_ITER: int = 10000
N_JOBS: int = -1
DIFFERENCE_BOUND: float = 0.01

# Analyzer
TRIAL_COUNT_DEFAULT: int = 3
TRIAL_MAX_ATTEMPTS: int = 5
LABEL_BIAS_RANGE_INTERVAL: float = 0.1
LABEL_BIAS_RANGE_MIN: float = 0.0
LABEL_BIAS_RANGE_MAX: float = 0.5


# Data Reader
ADULT_PARAMS = (
    {
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
    './Data/Adult/adult.data',
    0,
    './Data/Adult/adult.test',
    1,
    'Target',
    ['Race', 'Sex'])
SMALLADULT_PARAMS = (
    {
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
    './Data/SmallAdult/adult.data',
    0,
    './Data/SmallAdult/adult.test',
    1,
    'Target',
    ['Race', 'Sex'])
DEBUG_PARAMS = (
    {
        'A': np.int64,
        'B': np.int64,
        'C': np.int64
    },
    './Data/Debug/debug.data',
    0,
    './Data/Debug/debug.test',
    1,
    'C',
    ['B']
)

# File Handler / Plotter
DATA_DIR: str = './Data'

RESULTS_DIR: str = './Results'
FLIP_RATE_FILE_NAME: str = 'flip_rate.csv'

COL_TRIAL: str = 'Trial'
COL_FLIPRATE: str = 'Flip Rate'
COL_POSITIVE: str = 'Positive'
COL_NEGATIVE: str = 'Negative'
COL_CONFIDENCE_THRESHOLD: str = 'Confidence Threshold'
COL_UNIFORM: str = 'Uniform'
COL_ACCURACY: str = 'Accuracy'
COL_DP_DIFFERENCE: str = 'DP Difference'

MODELS = {'Unconstrained': 'red', 'DP Constrained': 'blue'}
METRICS = [COL_ACCURACY, COL_DP_DIFFERENCE]
