import numpy as np
import sys
import os
from dotenv import load_dotenv

load_dotenv()

CONTROL_PATH = os.getenv("CONTROL_PATH")
sys.path.append(CONTROL_PATH)

from t2_control_limit import calculate_t_square
import matplotlib.pyplot as plt


def evaluate_average(data: np.array) -> np.array:

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

    averages = np.average(di_chart, axis=0)
    print(averages)
    x_label = ["x1", "x2", "x3", "x4", "x5"]
    plt.bar(x_label, averages)
    plt.xticks(x_label)
    plt.show()

    return "Completed"
