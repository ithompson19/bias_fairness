from typing import List, Tuple
import os
import pandas as pd
import constants as const
from data_reader import DataReader

def generate_result_row(data_reader: DataReader, trial_num: int, flip_rate: Tuple[str, str, float, float], confidence_threshold: float, model_metrics: List[float]) -> pd.DataFrame:
    if len(model_metrics) != len(const.MODELS) * len(const.METRICS):
        raise ValueError('model_metrics array must include every combination of model and metric.')
    df: pd.DataFrame = pd.DataFrame(columns=__generate_column_names(data_reader))
    values = [trial_num]
    for column_name in data_reader.sensitive_attribute_column_names:
        for value in data_reader.sensitive_attribute_vals(column_name):
            if flip_rate[0]:
                if flip_rate[1].startswith('-'):
                    values.append(flip_rate[2] if flip_rate[0] == column_name and flip_rate[1].lstrip('-') != value else 0.0)
                    values.append(flip_rate[3] if flip_rate[0] == column_name and flip_rate[1].lstrip('-') != value else 0.0)
                else:
                    values.append(flip_rate[2] if flip_rate[0] == column_name and flip_rate[1] == value else 0.0)
                    values.append(flip_rate[3] if flip_rate[0] == column_name and flip_rate[1] == value else 0.0)
            else:
                values.append(flip_rate[2])
                values.append(flip_rate[3])
    values.append(confidence_threshold)
    values.extend(model_metrics)
    df.loc[0] = values
    return df

def save_results(data: pd.DataFrame, range_min: Tuple[str, str, float, float], range_max: Tuple[str, str, float, float], confidence_interval_test_flip_rate: Tuple[str, str, float, float]):
    file_name: str = f'{__generate_file_prefix(range_min, range_max, confidence_interval_test_flip_rate)}{const.FLIP_RATE_FILE_NAME}'
    try:
        os.makedirs(const.RESULTS_DIR)
    except FileExistsError:
        pass
    file_path = os.path.join(const.RESULTS_DIR, file_name)
    if os.path.exists(file_path):
        __move_results_to_new_subdirectory()
    print(f'Writing results to {file_path}')
    data.to_csv(file_path, index=False)

def read_all_results() -> List[Tuple[pd.DataFrame, str]]:
    try:
        os.makedirs(const.RESULTS_DIR)
    except FileExistsError:
        pass
    results_list: List[pd.DataFrame] = []
    for f in os.listdir(const.RESULTS_DIR):
        if f.endswith(const.FLIP_RATE_FILE_NAME):
            results_list.append((pd.read_csv(os.path.join(const.RESULTS_DIR, f)), f))
    return results_list

def generate_x_axis_name(results_file_name: str) -> str:
    keywords: List[str] = results_file_name.split('_')
    keywords[-1] = keywords[-1][:-len('.csv')]
    for i, keyword in enumerate(keywords):
        if keyword == const.COL_CONFIDENCE_THRESHOLD.replace(' ', '-').lower():
            keywords[i] = const.COL_CONFIDENCE_THRESHOLD
        elif '-' in keyword:
            keywords[i] = '-'.join([subword.capitalize() for subword in keyword.split('-')])
        else:
            keywords[i] = keyword.capitalize()
    if keywords[0] == const.COL_CONFIDENCE_THRESHOLD:
        return f'{keywords[0]} ({" ".join(keywords[1:])})'
    return ' '.join(keywords)

def generate_x_column(data: pd.DataFrame) -> str:
    if len(data[const.COL_CONFIDENCE_THRESHOLD].unique()) > 1:
        return const.COL_CONFIDENCE_THRESHOLD
    for column in data.columns:
        if column.endswith(const.COL_FLIPRATE) and len(data[column].unique()) > 1:
            return column
    raise ValueError('No columns have changed value in the data. Unable to plot.')

def generate_plot_file_name(results_file_name: str, metric_name: str) -> str:
    return os.path.join(const.RESULTS_DIR, f'{results_file_name[:-len(".csv")]}_{metric_name.lower()}.png')

def __generate_column_names(data_reader: DataReader) -> List[str]:
    columns: List[str] = [const.COL_TRIAL]
    for column_name in data_reader.sensitive_attribute_column_names:
        for value in data_reader.sensitive_attribute_vals(column_name):
            columns.append(f'{value} {const.COL_POSITIVE} {const.COL_FLIPRATE}')
            columns.append(f'{value} {const.COL_NEGATIVE} {const.COL_FLIPRATE}')
    columns.append(const.COL_CONFIDENCE_THRESHOLD)
    for model in const.MODELS:
        for test in const.METRICS:
            columns.append(f'{model} {test}')
    return columns

def __generate_file_prefix(range_min: float | Tuple[str, str, float, float],
                           range_max: float | Tuple[str, str, float, float],
                           confidence_interval_test_flip_rate: Tuple[str, str, float, float]) -> str:
    prefix: str = ''
    if isinstance(range_min, float):
        prefix = const.COL_CONFIDENCE_THRESHOLD.replace(' ', '-')
        if confidence_interval_test_flip_rate[0]:
            prefix = f'{prefix}_{"Non" if confidence_interval_test_flip_rate[1].startswith("-") else ""}{confidence_interval_test_flip_rate[1]}'
        if confidence_interval_test_flip_rate[3] == 0:
            prefix = f'{prefix}_{const.COL_POSITIVE}'
        elif confidence_interval_test_flip_rate[2] == 0:
            prefix = f'{prefix}_{const.COL_NEGATIVE}'
        if prefix == const.COL_CONFIDENCE_THRESHOLD.replace(' ', '-'):
            prefix = f'{prefix}_{const.COL_UNIFORM}'
    else:
        if range_min[0]:
            prefix = f'{"Non" if range_min[1].startswith("-") else ""}{range_min[1]}'
        if range_min[2] < range_max[2] and range_min[3] == range_max[3]:
            prefix = f'{prefix}{"_" if prefix else ""}{const.COL_POSITIVE}'
        elif range_min[3] < range_max[3] and range_min[2] == range_max[2]:
            prefix = f'{prefix}{"_" if prefix else ""}{const.COL_NEGATIVE}'
        if not prefix:
            prefix = const.COL_UNIFORM
    prefix = prefix + '_'
    prefix = prefix.lower()
    return prefix

def __move_results_to_new_subdirectory():
    min_unused_folder = 1
    while True:
        new_path = os.path.join(const.RESULTS_DIR, f'Results_{min_unused_folder}')
        try:
            os.makedirs(new_path)
            break
        except FileExistsError:
            min_unused_folder += 1
    print(f'Moving existing results to {new_path}')
    for f in os.listdir(const.RESULTS_DIR):
        if os.path.isfile(os.path.join(const.RESULTS_DIR, f)):
            os.rename(os.path.join(const.RESULTS_DIR, f), os.path.join(new_path, f))
