import numpy as np
import sys
import os
from dotenv import load_dotenv

load_dotenv()

CONTROL_PATH = os.getenv("CONTROL_PATH")
sys.path.append(CONTROL_PATH)

from t2_control_limit import calculate_t_square
import matplotlib.pyplot as plt


def evaluate_ranksum(data: np.array) -> np.array:

    t2_values = calculate_t_square(data, mode="single")
    t2_values = np.array(t2_values)

    num_variables = data.shape[1]

    di_chart = []
    for column in range(num_variables):

        t_i = data[:, column]
        t_i2 = np.multiply(t_i, t_i)

        d_i = np.abs(np.subtract(t2_values, t_i2))
        di_chart.append(d_i)

    di_chart = np.array(di_chart).T

    rank_chart = []
    for row in range(di_chart.shape[0]):

        ranked_one_row = di_chart[row, :].argsort()
        rank_chart.append(ranked_one_row)

    rank_chart = np.array(rank_chart)
    rank_addition = np.sum(rank_chart, axis=0)
    rank_addition = np.array(rank_addition)
    norm_rank_addition = np.divide(rank_addition, np.sum(rank_addition, axis=0))
    print(norm_rank_addition)

    x_label = ["x1", "x2", "x3", "x4", "x5"]
    plt.bar(x_label, norm_rank_addition)
    plt.xticks(x_label)
    plt.show()

    return "Completed"
