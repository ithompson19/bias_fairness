from typing import List, Tuple
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
from data_reader import DataReader
import file_handler
import constants as const
import tests_generator

# TODO: remove
def plot_group_old(title: str, plots: List[List[Axes]], y_axis: str, x_axis_names: List[str], x_axes_units: List[str]):
    plt.close()
    plt.ion()
    plt.figure()
    fig, axs = plt.subplots(nrows=len(plots), ncols=len(plots[0]), figsize=(12, 9))
    plt.suptitle(title)
    for i, axes_row in enumerate(plots):
        for j, ax in enumerate(axes_row):
            for line in ax.lines:
                axs[i, j].plot(*line.get_data())
                axs[i, j].set(xlabel=f'{x_axis_names[i]} {x_axes_units[j]}', ylabel=y_axis)
    leg = fig.legend(labels=const.MODEL_LINES, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=len(const.MODEL_LINES))
    for i, color in enumerate(const.MODEL_LINES.values()):
        leg.legendHandles[i].set_color(color)
    plt.tight_layout()
    plt.show()
    plt.ioff()

def plot_accuracy_fairness(data_reader: DataReader, tests: List[Tuple[Tuple[str, str, float, float], float]]):
    print('Generating figures...')
    plt.ioff()
    data, file_name = file_handler.read_metrics(data_reader, tests)
    __plot_accuracy(data = data, file_name = file_name)
    __plot_fairness(data = data, file_name = file_name)
    plt.ion()

def plot_group_accuracy_fairness(title: str, data_reader: DataReader, sensitive_attribute_column: str, is_advantaged: bool = None):
    print('Generating group figures...')
    __plot_group_accuracy(title, data_reader, sensitive_attribute_column, is_advantaged)
    __plot_group_fairness(title, data_reader, sensitive_attribute_column, is_advantaged)
        
def __plot_accuracy(data: pd.DataFrame, file_name: str):
    plt.close()
    __plot(title = 'Accuracy',
           y_label = 'Accuracy',
           y_columns = [f'{model} {const.COL_ACCURACY}' for model in const.MODEL_LINES],
           fairness_plot = False,
           data = data,
           file_name = file_name)

def __plot_fairness(data: pd.DataFrame, file_name: str):
    plt.close()
    __plot(title = 'Fairness Violation',
           y_label = 'Fairness Violation',
           y_columns = [model_metric_name for (_, model_metric_name) in const.CONSTRAINED_MODELS.values()],
           fairness_plot = True,
           data = data,
           file_name = file_name)

def __plot_group_accuracy(title: str, data_reader: DataReader, sensitive_attribute_column: str, is_advantaged: bool = None):
    plt.figure()
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))
    plt.ioff()
    for i, tests_row in enumerate(tests_generator.generate_dummy_tests_for_group(data_reader, sensitive_attribute_column, is_advantaged)):
        for j, tests in enumerate(tests_row):
            data, file_name = file_handler.read_metrics(data_reader, tests)
            __plot(title='',
                y_label='Accuracy',
                y_columns=[f'{model} {const.COL_ACCURACY}' for model in const.MODEL_LINES],
                fairness_plot=False,
                data=data,
                file_name=file_name,
                ax=ax[i, j])
    plt.ion()
    fig.suptitle(title)
    leg = fig.legend(labels=const.MODEL_LINES, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=len(const.MODEL_LINES))
    for i, color in enumerate(const.MODEL_LINES.values()):
        leg.legendHandles[i].set_color(color)
    plt.tight_layout()
    plt.show()
    plt.ioff()

def __plot_group_fairness(title: str, data_reader: DataReader, sensitive_attribute_column: str, is_advantaged: bool = None):
    plt.figure()
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))
    plt.ioff()
    for i, tests_row in enumerate(tests_generator.generate_dummy_tests_for_group(data_reader, sensitive_attribute_column, is_advantaged)):
        for j, tests in enumerate(tests_row):
            data, file_name = file_handler.read_metrics(data_reader, tests)
            __plot(title='',
                y_label='Fairness Violation',
                y_columns=[model_metric_name for (_, model_metric_name) in const.CONSTRAINED_MODELS.values()],
                fairness_plot=True,
                data=data,
                file_name=file_name,
                ax=ax[i, j])
    plt.ion()
    fig.suptitle(title)
    leg = fig.legend(labels=const.MODEL_LINES, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=len(const.MODEL_LINES))
    for i, color in enumerate(const.MODEL_LINES.values()):
        leg.legendHandles[i].set_color(color)
    plt.tight_layout()
    plt.show()
    plt.ioff()

def __plot(title: str, y_label: str, y_columns: List[str], fairness_plot: bool, data: pd.DataFrame, file_name: str, ax = None):
    x_axis_name = file_handler.generate_figure_x_axis_name(file_name)
    x_column_name = file_handler.find_figure_x_column(data)
    
    plt.figure()
    is_file_plot = False
    if ax is None:
        plt.suptitle(title)
        _, ax = plt.subplots()
        is_file_plot = True
    ax.set(xlabel=x_axis_name, ylabel=y_label)

    grouped_data = data.groupby(x_column_name, as_index=False)[y_columns].agg({y_col: ['mean', 'std'] for y_col in y_columns})
    
    for (label, color), y_col in zip(list(const.MODEL_LINES.items())[int(fairness_plot):], y_columns):
        ax.fill_between(grouped_data[x_column_name],
                        grouped_data[y_col]['mean'] - grouped_data[y_col]['std'],
                        grouped_data[y_col]['mean'] + grouped_data[y_col]['std'],
                        color=mpl.colors.to_rgba(color, 0.15))
        ax.plot(grouped_data[x_column_name], grouped_data[y_col]['mean'], color=color, label=label)
    if is_file_plot:
        ax.legend(bbox_to_anchor=(1.04, 0.5),
              loc='center left',
              borderaxespad=0)
        file_handler.save_figure(file_name, fairness_plot)
