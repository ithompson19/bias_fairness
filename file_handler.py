from typing import Dict, List, Tuple
import constants as const
import pandas as pd
import os

def generate_result_row(trial_num: int, flip_rate: Dict[str, Dict[str, Tuple[float, float]]], confidence_threshold: float, model_metrics: List[float]) -> pd.DataFrame:
    if len(model_metrics) != len(const.MODELS) * len(const.METRICS):
        raise ValueError('model_metrics array must include every combination of model and metric.')
    df: pd.DataFrame = pd.DataFrame(columns=__generate_column_names(flip_rate))
    values = [trial_num]
    for sensitive_attribute_name, sensitive_attribute_values in flip_rate.items():
        for sensitive_attribute_value in sensitive_attribute_values:
            values.append(flip_rate[sensitive_attribute_name][sensitive_attribute_value][0])
            values.append(flip_rate[sensitive_attribute_name][sensitive_attribute_value][1])
    values.append(confidence_threshold)
    values.extend(model_metrics)
    df.loc[0] = values
    return df
    
def save_results_to_file(data: pd.DataFrame):
    try:
        os.makedirs(const.RESULTS_DIR)
    except:
        pass
    if os.path.isfile(const.RESULTS_FILE_LOC):
        __move_results_to_new_subdirectory()
    
    print(f'Writing results to {const.RESULTS_FILE_LOC}')
    data.to_csv(const.RESULTS_FILE_LOC, index=False)
    
def read_from_subdirectory(subdirectory: str = '') -> pd.DataFrame:
    path: str = os.path.join(const.RESULTS_DIR, subdirectory, const.RESULTS_FILE_NAME) if subdirectory else const.RESULTS_FILE_LOC
    try:
        return pd.read_csv(path)
    except:
        raise IOError(f'{path} cannot be read.')
    
def full_path_from_subdirectory(subdirectory: str, file_name: str) -> str:
    path: str = os.path.join(const.RESULTS_DIR, subdirectory, file_name) if subdirectory else os.path.join(const.RESULTS_DIR, file_name)
    return path

def __generate_column_names(flip_rate: Dict[str, Dict[str, Tuple[float, float]]]) -> List[str]:
    columns: List[str] = [const.COL_TRIAL]
    for sensitive_attribute_values in flip_rate.values():
        for sensitive_attribute_value in sensitive_attribute_values:
            columns.append(f'{sensitive_attribute_value} {const.COL_POSITIVE} {const.COL_FLIPRATE}')
            columns.append(f'{sensitive_attribute_value} {const.COL_NEGATIVE} {const.COL_FLIPRATE}')
    columns.append(const.COL_CONFIDENCE_THRESHOLD)
    for model in const.MODELS.keys():
        for test in const.METRICS:
            columns.append(f'{model} {test}')
    return columns

def __move_results_to_new_subdirectory():
    min_unused_folder = 1
    while True:
        new_path = os.path.join(const.RESULTS_DIR, f'Results_{min_unused_folder}')
        try:
            os.makedirs(new_path)
            break
        except:
            min_unused_folder += 1
    print(f'Moving existing results to {new_path}')
    for f in os.listdir(const.RESULTS_DIR):
        if os.path.isfile(os.path.join(const.RESULTS_DIR, f)):
            os.rename(os.path.join(const.RESULTS_DIR, f), os.path.join(new_path, f))