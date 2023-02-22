"""Holds all constants.
"""
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, true_positive_rate, false_positive_rate
from fairlearn.reductions import DemographicParity, EqualizedOdds, FalsePositiveRateParity, TruePositiveRateParity

# ML Models
MAX_ITER: int = 10000
N_JOBS: int = -1
DIFFERENCE_BOUND: float = 0.01

# Analyzer
TRIAL_COUNT_DEFAULT: int = 10
TRIAL_MAX_ATTEMPTS: int = 5
LABEL_BIAS_RANGE_INTERVAL: float = 0.1


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
    ('<=50K', '>50K'),
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
    ('<=50K', '>50K'),
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
    (0, 1),
    ['B']
)

# File Handler / Plotter
DIR_DATA: str = './Data'
DIR_MODELS: str = './Models'
DIR_METRICS: str = './Metrics'
DIR_FIGURES: str = './Figures'

FILE_NAME_MODELS: str = 'trained_models.pkl'
FILE_NAME_METRICS: str = 'metrics.csv'
FILE_NAME_FIGURES_ACCURACY: str = 'accuracy.png'
FILE_NAME_FIGURES_FAIRNESS: str = 'fairness.png'

COL_TRIAL: str = 'Trial'
COL_FLIPRATE: str = 'Flip Rate'
COL_POSITIVE: str = 'Positive'
COL_NEGATIVE: str = 'Negative'
COL_CONFIDENCE_THRESHOLD: str = 'Confidence Threshold'
COL_UNIFORM: str = 'Uniform'

COL_ACCURACY: str = 'Accuracy'
COL_DP_DIFFERENCE: str = 'Demographic Parity Difference'
COL_EO_DIFFERENCE: str = 'Equalized Odds Difference'
COL_TRUE_POS_RATE: str = 'True Positive Rate'
COL_FALSE_POS_RATE: str = 'False Positive Rate'

CONSTRAINED_MODELS = {DemographicParity: (demographic_parity_difference, COL_DP_DIFFERENCE),
                      EqualizedOdds: (equalized_odds_difference, COL_EO_DIFFERENCE),
                      TruePositiveRateParity: (true_positive_rate, COL_TRUE_POS_RATE),
                      FalsePositiveRateParity: (false_positive_rate, COL_FALSE_POS_RATE)}
MODEL_LINES = {'Unconstrained': 'tab:red',
               'Demographic Parity': 'tab:blue',
               'Equalized Odds': 'tab:green',
               'Equality of Opportunity': 'tab:purple',
               'False Positive Rate Parity': 'tab:orange'}
