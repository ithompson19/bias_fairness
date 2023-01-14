from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def plot_batch(title: str, x_label: str, y_label: str, batch: Dict[str, Tuple[List[float], List[float]]]):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for line_label, line_values in batch.items():
        plt.plot(line_values[0], line_values[1], label = line_label)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')