import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import file_handler
import constants as const

def plot_all():
    results_list = file_handler.read_all_results()
    for data, file_name in results_list:
        for metric in const.METRICS:
            __plot(metric_name=metric, data=data, file_name=file_name)

def __plot(metric_name: str, data: pd.DataFrame, file_name: str):
    x_axis_name = file_handler.generate_x_axis_name(file_name)
    x_column_name = file_handler.generate_x_column(data)
    
    plt.figure()
    plt.suptitle(f'{metric_name} of Fairness Constrained and Unconstrained ML Models')
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
        
    ax.legend(bbox_to_anchor=(1.04, 0.5),
              loc='center left',
              borderaxespad=0)
    plt.savefig(file_handler.generate_plot_file_name(file_name, metric_name), bbox_inches='tight')
