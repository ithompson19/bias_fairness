from typing import List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from itertools import islice
import file_handler
import constants as const

def plot_all(tests: List[Tuple[Tuple[str, str, float, float], float]]):
    print('Generating figures...')
    data, file_name = file_handler.read_metrics(tests)
    __plot_accuracy(data = data, file_name = file_name)
    __plot_fairness(data = data, file_name = file_name)
        
def __plot_accuracy(data: pd.DataFrame, file_name: str):
    __plot(title = 'Accuracy',
           y_label = 'Accuracy',
           y_columns = [f'{model} {const.COL_ACCURACY}' for model in const.MODEL_LINES],
           fairness_plot = False,
           data = data,
           file_name = file_name)

def __plot_fairness(data: pd.DataFrame, file_name: str):
    __plot(title = 'Fairness Metrics',
           y_label = 'Fairness Violation',
           y_columns = [model_metric_name for (_, model_metric_name) in const.CONSTRAINED_MODELS.values()],
           fairness_plot = True,
           data = data,
           file_name = file_name)

def __plot(title: str, y_label: str, y_columns: List[str], fairness_plot: bool, data: pd.DataFrame, file_name: str):
    x_axis_name = file_handler.generate_figure_x_axis_name(file_name)
    x_column_name = file_handler.find_figure_x_column(data)
    
    plt.figure()
    plt.suptitle(title)
    _, ax = plt.subplots()
    ax.set(xlabel=x_axis_name, ylabel=y_label)

    grouped_data = data.groupby(x_column_name, as_index=False)[y_columns].agg({y_col: ['mean', 'std'] for y_col in y_columns})
    
    for (label, color), y_col in zip(list(const.MODEL_LINES.items())[int(fairness_plot):], y_columns):
        ax.fill_between(grouped_data[x_column_name],
                        grouped_data[y_col]['mean'] - grouped_data[y_col]['std'],
                        grouped_data[y_col]['mean'] + grouped_data[y_col]['std'],
                        color=mpl.colors.to_rgba(color, 0.15))
        plt.plot(grouped_data[x_column_name], grouped_data[y_col]['mean'], color=color, label=label)
        
    ax.legend(bbox_to_anchor=(1.04, 0.5),
              loc='center left',
              borderaxespad=0)
    file_handler.save_figure(file_name, fairness_plot)
