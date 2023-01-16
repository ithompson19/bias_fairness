from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_batch(title: str, x_label: str, y_label: str, batch: Dict[str, Tuple[List[float], List[float]]]):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for line_label, line_values in batch.items():
        plt.plot(line_values[0], line_values[1], label = line_label)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')

def plot_all(data: pd.DataFrame = pd.read_csv('./Results/results.csv')):
    plot_accuracy_flip_rate(data)
    plot_dp_flip_rate(data)
    
def plot_accuracy_flip_rate(data: pd.DataFrame = pd.read_csv('./Results/results.csv')):
    plt.figure()
    plt.title('Accuracy of Fairness Constrained and Unconstrained ML Models')
    plt.xlabel('Label Flip rate')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    fig, ax = plt.subplots()
    grouped_data = data.groupby('Flip Rate', as_index=False)[['Unconstrained Accuracy', 'DP Constrained Accuracy']].agg({
        'Unconstrained Accuracy':['min', 'max', 'mean'],
        'DP Constrained Accuracy':['min', 'max', 'mean']})
    
    ax.fill_between(grouped_data['Flip Rate'], 
                    grouped_data['Unconstrained Accuracy']['min'], 
                    grouped_data['Unconstrained Accuracy']['max'], 
                    color=mpl.colors.to_rgba('red', 0.15))
    plt.plot(grouped_data['Flip Rate'], grouped_data['Unconstrained Accuracy']['mean'], color='red', label='Unconstrained')
    ax.fill_between(grouped_data['Flip Rate'], 
                    grouped_data['DP Constrained Accuracy']['min'], 
                    grouped_data['DP Constrained Accuracy']['max'], 
                    color=mpl.colors.to_rgba('blue', 0.15))
    plt.plot(grouped_data['Flip Rate'], grouped_data['DP Constrained Accuracy']['mean'], color='blue', label='DP Constrained')
    
    plt.legend()
    plt.savefig('./Results/accuracy_flip_rate.png')
    
def plot_dp_flip_rate(data: pd.DataFrame = pd.read_csv('./Results/results.csv')):
    plt.figure()
    plt.title('Demographic Parity of Fairness Constrained and Unconstrained ML Models')
    plt.xlabel('Label Flip rate')
    plt.ylabel('Demographic Parity Ratio')
    plt.ylim(0, 1)
    
    fig, ax = plt.subplots()
    grouped_data = data.groupby('Flip Rate', as_index=False)[['Unconstrained DP Ratio', 'DP Constrained DP Ratio']].agg({
        'Unconstrained DP Ratio':['min', 'max', 'mean'],
        'DP Constrained DP Ratio':['min', 'max', 'mean']})
    
    ax.fill_between(grouped_data['Flip Rate'], 
                    grouped_data['Unconstrained DP Ratio']['min'], 
                    grouped_data['Unconstrained DP Ratio']['max'], 
                    color=mpl.colors.to_rgba('red', 0.15))
    plt.plot(grouped_data['Flip Rate'], grouped_data['Unconstrained DP Ratio']['mean'], color='red', label='Unconstrained')
    ax.fill_between(grouped_data['Flip Rate'], 
                    grouped_data['DP Constrained DP Ratio']['min'], 
                    grouped_data['DP Constrained DP Ratio']['max'], 
                    color=mpl.colors.to_rgba('blue', 0.15))
    plt.plot(grouped_data['Flip Rate'], grouped_data['DP Constrained DP Ratio']['mean'], color='blue', label='DP Constrained')
    
    plt.legend()
    plt.savefig('./Results/dp_flip_rate.png')
    
plot_all()