import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import file_handler
import constants as const


def plot_all(subdirectory: str = ''):
    data: pd.DataFrame = file_handler.read_from_subdirectory(subdirectory)
    plot_fliprate_accuracy(subdirectory, data)
    plot_fliprate_dp(subdirectory, data)
    
def plot_fliprate_accuracy(subdirectory: str = '', data: pd.DataFrame = None):
    __plot(title='Accuracy', x='Flip Rate', y='Accuracy', subdirectory=subdirectory, data=data)
    
def plot_fliprate_dp(subdirectory: str = '', data: pd.DataFrame = None):
    __plot(title='Demographic Parity', x='Flip Rate', y='DP Ratio', subdirectory=subdirectory, data=data)
    
def __plot(title: str, x: str, y: str, subdirectory: str, data: pd.DataFrame = None):
    if data.empty:
        data = file_handler.read_from_subdirectory(subdirectory)
    
    plt.figure()
    plt.title(f'{title} of Fairness Constrained and Unconstrained ML Models')
    plt.xlabel(x)
    plt.ylabel(y)
    _, ax = plt.subplots()
    
    keys = [f'{key} {y}' for key in const.LINES]
            
    grouped_data = data.groupby(x, as_index=False)[keys].agg({key: ['min', 'max', 'mean'] for key in keys})
    
    for label, color in const.LINES.items():
        ax.fill_between(grouped_data[x], 
                        grouped_data[f'{label} {y}']['min'], 
                        grouped_data[f'{label} {y}']['max'], 
                        color=mpl.colors.to_rgba(color, 0.15))
        plt.plot(grouped_data[x], grouped_data[f'{label} {y}']['mean'], color=color, label=label)
        
    plt.legend()
    plt.savefig(file_handler.full_path_from_subdirectory(subdirectory, f'{x.replace(" ", "").lower()}_{y.replace(" ", "").lower()}.png'))