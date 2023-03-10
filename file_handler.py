import dill
from typing import List, Tuple
import os
import pandas as pd
import constants as const
import matplotlib.pyplot as plt
from data_reader import DataReader

# MODELS

def prepare_model_directory(tests: List[Tuple[Tuple[str, str, float, float], float]]):
    try:
        os.makedirs(const.DIR_MODELS)
    except FileExistsError:
        pass
    dir_name: str = os.path.join(const.DIR_MODELS, __generate_file_prefix(tests))
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        if len(os.listdir(dir_name)) > 0:
            min_unused_dir_num = 1
            new_dir_name: str = f'{dir_name} ({min_unused_dir_num})'
            while os.path.isfile(new_dir_name):
                min_unused_dir_num += 1
                new_dir_name: str = f'{dir_name} ({min_unused_dir_num})'
            print(f'Moving existing directory to {new_dir_name}')
            os.rename(dir_name, new_dir_name)
            os.makedirs(dir_name)

def save_models(tests: List[Tuple[Tuple[str, str, float, float], float]], trial_num: int, models: list):
    file_name: str = __get_models_file_name(tests, trial_num)
    models_file = open(file_name, 'wb')
    dill.dump(models, models_file)
    models_file.close()

def read_models(tests: List[Tuple[Tuple[str, str, float, float], float]]) -> list:
    dir_name: str = os.path.join(const.DIR_MODELS, __generate_file_prefix(tests))
    if not os.path.isdir(dir_name):
        raise FileNotFoundError('Directory corresponding to provided tests cannot be found.')
    file_names: List[Tuple[int, str]] = __get_file_names_from_model_directory(dir_name)
    if len(file_names) == 0:
        raise FileNotFoundError('No trained models exist in directory.')
    models = []
    for trial_num, file_name in file_names:
        models_file = open(file_name, 'rb')
        trained_models = dill.load(models_file)
        models.append((trial_num, trained_models))
        models_file.close()
    return models
    
# METRICS

def generate_metrics_row(data_reader: DataReader, trial_num: int, flip_rate: Tuple[str, str, float, float], confidence_threshold: float, model_metrics: List[float]) -> pd.DataFrame:
    if len(model_metrics) != len(const.MODEL_LINES) + len(const.CONSTRAINED_MODELS):
        raise ValueError('model_metrics array must include an accuracy measurement for every model, and the appropriate metric for constrained models.')
    df: pd.DataFrame = pd.DataFrame(columns=__generate_column_names(data_reader, flip_rate))
    values = [trial_num]
    for value in data_reader.sensitive_attribute_vals(flip_rate[0]):
        if flip_rate[1]:
            if flip_rate[1].startswith('-'):
                values.append(flip_rate[2] if flip_rate[1].lstrip('-') != value else 0.0)
                values.append(flip_rate[3] if flip_rate[1].lstrip('-') != value else 0.0)
            else:
                values.append(flip_rate[2] if flip_rate[1] == value else 0.0)
                values.append(flip_rate[3] if flip_rate[1] == value else 0.0)
        else:
            values.append(flip_rate[2])
            values.append(flip_rate[3])
    values.append(confidence_threshold)
    values.extend(model_metrics)
    df.loc[0] = values
    return df

def save_metrics(tests: List[Tuple[Tuple[str, str, float, float], float]], metrics: pd.DataFrame):
    metrics.to_csv(__get_metrics_file_name(tests, True))
    
def read_metrics(tests: List[Tuple[Tuple[str, str, float, float], float]]) -> Tuple[pd.DataFrame, str]:
    if not os.path.isdir(const.DIR_METRICS):
        raise FileNotFoundError('Metrics directory does not exist. Models must be analyzed first.')
    file_name = __get_metrics_file_name(tests, False)
    return pd.read_csv(file_name), file_name

# FIGURES

def generate_figure_x_axis_name(metrics_file_name: str) -> str:
    keywords: List[str] = os.path.basename(metrics_file_name).split('_')
    keywords = keywords[:-1]
    for i, keyword in enumerate(keywords):
        if keyword == const.COL_CONFIDENCE_THRESHOLD.replace(' ', '-').lower():
            keywords[i] = const.COL_CONFIDENCE_THRESHOLD
        else:
            keywords[i] = '-'.join([subword.capitalize() for subword in keyword.split('-')])
    if keywords[0] == const.COL_CONFIDENCE_THRESHOLD:
        return f'{keywords[0]} ({" ".join(keywords[1:])})'
    return ' '.join(keywords)

def find_figure_x_column(data: pd.DataFrame) -> str:
    if len(data[const.COL_CONFIDENCE_THRESHOLD].unique()) > 1:
        return const.COL_CONFIDENCE_THRESHOLD
    for column in data.columns:
        if column.endswith(const.COL_FLIPRATE) and len(data[column].unique()) > 1:
            return column
    raise ValueError('No columns have changed value in the data. Unable to plot.')

def save_figure(metrics_file_name: str, fairness_plot: bool):
    plt.savefig(__get_figure_file_name(metrics_file_name, fairness_plot), bbox_inches = 'tight')

# HELPERS (MODELS)

def __get_models_file_name(tests: List[Tuple[Tuple[str, str, float, float], float]], trial_num: int) -> str:
    split_name = const.FILE_NAME_MODELS.split('.')
    file_name = f'{split_name[0]}_{trial_num}.{split_name[1]}'
    return os.path.join(const.DIR_MODELS, __generate_file_prefix(tests), file_name)

def __get_file_names_from_model_directory(dir_name: str) -> List[Tuple[int, str]]:
    file_names: List[Tuple[int, str]] = []
    for f in os.listdir(dir_name):
        file_names.append((int(f.split('.')[0][-1]), os.path.join(dir_name, f)))
    return file_names

# HELPERS (METRICS)

def __get_metrics_file_name(tests: List[Tuple[Tuple[str, str, float, float], float]], move: bool) -> str:
    try:
        os.makedirs(const.DIR_METRICS)
    except FileExistsError:
        pass
    file_name: str = os.path.join(const.DIR_METRICS, f'{__generate_file_prefix(tests)}_{const.FILE_NAME_METRICS}')
    if move:
        __move_file_if_exists(file_name)
    return file_name

def __generate_column_names(data_reader: DataReader, flip_rate: Tuple[str, str, float, float]) -> List[str]:
    columns: List[str] = [const.COL_TRIAL]
    for value in data_reader.sensitive_attribute_vals(flip_rate[0]):
        columns.append(f'{value} {const.COL_POSITIVE} {const.COL_FLIPRATE}')
        columns.append(f'{value} {const.COL_NEGATIVE} {const.COL_FLIPRATE}')
    columns.append(const.COL_CONFIDENCE_THRESHOLD)
    for model in const.MODEL_LINES:
        columns.append(f'{model} {const.COL_ACCURACY}')
    for (_, model_metric_name) in const.CONSTRAINED_MODELS.values():
        columns.append(model_metric_name)
    return columns

# HELPERS (FIGURES)

def __get_figure_file_name(metrics_file_name: str, fairness_plot: bool) -> str:
    try:
        os.makedirs(const.DIR_FIGURES)
    except FileExistsError:
        pass
    metrics_file_name = metrics_file_name[len(const.DIR_METRICS) + 1:]
    metrics_file_name = metrics_file_name[:-len(const.FILE_NAME_METRICS) - 1]
    file_name = os.path.join(const.DIR_FIGURES, f'{metrics_file_name}_{const.FILE_NAME_FIGURES_FAIRNESS if fairness_plot else const.FILE_NAME_FIGURES_ACCURACY}')
    __move_file_if_exists(file_name)
    return file_name

# HELPERS (GENERIC)

def __generate_file_prefix(tests: List[Tuple[Tuple[str, str, float, float], float]]) -> str:
    if len(tests) < 2:
        raise ValueError('File name cannot be made from empty or singular list of tests.')
    prefix: str = ''
    if len(set([arg[1] for arg in tests])) > 1:
        prefix = const.COL_CONFIDENCE_THRESHOLD.replace(' ', '-')
        if tests[0][0][0]:
            prefix = f'{prefix}_{"non" if tests[0][0][1].startswith("-") else ""}{tests[0][0][1]}'
        if tests[0][0][3] == 0:
            prefix = f'{prefix}_{const.COL_POSITIVE}'
        elif tests[0][0][2] == 0:
            prefix = f'{prefix}_{const.COL_NEGATIVE}'
        if prefix == const.COL_CONFIDENCE_THRESHOLD.replace(' ', '-'):
            prefix = f'{prefix}_{const.COL_UNIFORM}'
    else:
        if tests[0][0][0]:
            prefix = f'{"non" if tests[0][0][1].startswith("-") else ""}{tests[0][0][1]}'
        if tests[0][0][2] < tests[-1][0][2] and tests[0][0][3] == tests[-1][0][3]:
            prefix = f'{prefix}{"_" if prefix else ""}{const.COL_POSITIVE}'
        elif tests[0][0][3] < tests[-1][0][3] and tests[0][0][2] == tests[-1][0][2]:
            prefix = f'{prefix}{"_" if prefix else ""}{const.COL_NEGATIVE}'
        if not prefix:
            prefix = const.COL_UNIFORM
    prefix = prefix.lower()
    return prefix

def __move_file_if_exists(file_name: str):
    if os.path.isfile(file_name):
        min_unused_file_num: int = 1
        base_name: str = os.path.basename(file_name)
        new_file_name: str = f'{base_name.split(".")[0]}_{min_unused_file_num}.{base_name.split(".")[1]}'
        new_file_name = os.path.join(file_name[:-len(base_name) - 1], new_file_name)
        while os.path.isfile(new_file_name):
            min_unused_file_num += 1
            base_name: str = os.path.basename(file_name)
            new_file_name: str = f'{base_name.split(".")[0]}_{min_unused_file_num}.{base_name.split(".")[1]}'
            new_file_name = os.path.join(file_name[:-len(base_name) - 1], new_file_name)
        print(f'Moving existing file to {new_file_name}')
        os.rename(file_name, new_file_name)
