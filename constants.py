import os
import numpy as np

# ML Models
MAX_ITER = 10000
N_JOBS = -1
DIFFERENCE_BOUND = 0.01

# Analyzer
TRIAL_COUNT_DEFAULT = 10
TRIAL_MAX_ATTEMPTS = 5
RESULT_COLUMNS = ['Flip Rate',
                  'Confidence Threshold', 
                  'Unconstrained Accuracy', 
                  'Unconstrained DP Ratio', 
                  'DP Constrained Accuracy', 
                  'DP Constrained DP Ratio']
LABEL_BIAS_RANGE_MIN = 0
LABEL_BIAS_RANGE_MAX = 0.5

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
POKEMON_PARAMS = (
    {
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
    './Data/Pokemon/pokemon.data',
    0,
    './Data/Pokemon/pokemon.test',
    1,
    'Legendary',
    ['Total', 'Generation'])
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

# File Handler
DATA_DIR = './Data'
RESULTS_DIR = './Results'
RESULTS_FILE_NAME = 'results.csv'
RESULTS_FILE_LOC = os.path.join(RESULTS_DIR, RESULTS_FILE_NAME)

# Plotter
LINES = {'Unconstrained': 'red', 'DP Constrained': 'blue'}