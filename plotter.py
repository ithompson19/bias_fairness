from typing import List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import file_handler
import constants as const


def plot_all(subdirectory: str = ''):
    data: pd.DataFrame = file_handler.read_from_subdirectory(subdirectory)
    plot_accuracy(subdirectory, data)
    plot_dp(subdirectory, data)
    
def plot_accuracy(subdirectory: str = '', data: pd.DataFrame = pd.DataFrame()):
    __plot(title='Accuracy', metric_name=const.COL_ACCURACY, subdirectory=subdirectory, data=data)
    
def plot_dp(subdirectory: str = '', data: pd.DataFrame = pd.DataFrame()):
    __plot(title='Demographic Parity', metric_name=const.COL_DP_DIFFERENCE, subdirectory=subdirectory, data=data)
    
def __plot(title: str, metric_name: str, subdirectory: str, data: pd.DataFrame = pd.DataFrame()):
    if data.empty:
        data = file_handler.read_from_subdirectory(subdirectory)
    
    x_axis_name, x_column_name = __get_x_axis_from_data(data)
    
    plt.figure()
    plt.suptitle(f'{title} of Fairness Constrained and Unconstrained ML Models')
    _, ax = plt.subplots()
    ax.set(xlabel=x_axis_name, ylabel=metric_name)
    
    keys = [f'{key} {metric_name}' for key in const.MODELS]
            
    grouped_data = data.groupby(x_column_name, as_index=False)[keys].agg({key: ['min', 'max', 'mean'] for key in keys})
    
    for label, color in const.MODELS.items():
        ax.fill_between(grouped_data[x_column_name], 
                        grouped_data[f'{label} {metric_name}']['min'], 
                        grouped_data[f'{label} {metric_name}']['max'], 
                        color=mpl.colors.to_rgba(color, 0.15))
        plt.plot(grouped_data[x_column_name], grouped_data[f'{label} {metric_name}']['mean'], color=color, label=label)
        
    plt.legend()
    plt.savefig(file_handler.full_path_from_subdirectory(subdirectory, f'{x_axis_name.replace(" ", "").lower()}_{metric_name.replace(" ", "").lower()}.png'))
    
def __get_x_axis_from_data(data: pd.DataFrame) -> Tuple[str, str]:
    x_columns_of_interest = __parse_x_columns(data)
    x_axis_name: str = const.COL_FLIPRATE
    x_column_name: str = f'{x_columns_of_interest[0][0]} {x_columns_of_interest[0][1]} {const.COL_FLIPRATE}'
    if x_columns_of_interest[0][0] == const.COL_CONFIDENCE_THRESHOLD:
        if len(x_columns_of_interest) > 1:
            raise ValueError('Multiple values changing in data. Cannot be graphed.')
        else:
            return const.COL_CONFIDENCE_THRESHOLD, const.COL_CONFIDENCE_THRESHOLD
    for i in range(len(x_columns_of_interest)):
        for j in range(i, len(x_columns_of_interest)):
            # Attribute name shared between pos/neg columns (exactly 2 columns)
            if x_columns_of_interest[i][0] == x_columns_of_interest[j][0]:
                if x_axis_name == const.COL_FLIPRATE:
                    x_axis_name = f'{x_columns_of_interest[i][0]} {const.COL_FLIPRATE}'
            # Pos/neg shared between columns (at least 2 columns)
            elif x_columns_of_interest[i][1] == x_columns_of_interest[j][1]:
                if x_axis_name == const.COL_FLIPRATE:
                    x_axis_name = f'{x_columns_of_interest[i][1]} {const.COL_FLIPRATE}'
    return x_axis_name, x_column_name

def __parse_x_columns(data: pd.DataFrame) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    x_columns_of_interest: List[str] = filter(lambda column: const.COL_FLIPRATE in column and len(data[column].unique()) > 1, data.loc[data[const.COL_TRIAL] == 1].columns)
    for x_col in x_columns_of_interest:
        pos_neg = const.COL_POSITIVE if const.COL_POSITIVE in x_col else const.COL_NEGATIVE
        attr_value = x_col[:x_col.index(pos_neg)].strip()
        results.append((attr_value, pos_neg))
    if len(data[const.COL_CONFIDENCE_THRESHOLD].unique()) > 1:
        results.append((const.COL_CONFIDENCE_THRESHOLD, ''))
    return results