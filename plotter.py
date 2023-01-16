from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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
    plt.plot(data['Flip Rate'], data['Unconstrained Accuracy'], label = 'Unconstrained')
    plt.plot(data['Flip Rate'], data['DP Constrained Accuracy'], label = 'DP Constrained')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig('./Results/accuracy_flip_rate.png')
    
def plot_dp_flip_rate(data: pd.DataFrame = pd.read_csv('./Results/results.csv')):
    plt.figure()
    plt.title('Demographic Parity of Fairness Constrained and Unconstrained ML Models')
    plt.xlabel('Label Flip rate')
    plt.ylabel('Demographic Parity Ratio')
    plt.plot(data['Flip Rate'], data['Unconstrained DP Ratio'], label = 'Unconstrained')
    plt.plot(data['Flip Rate'], data['DP Constrained DP Ratio'], label = 'DP Constrained')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig('./Results/dp_flip_rate.png')
    
plot_all()