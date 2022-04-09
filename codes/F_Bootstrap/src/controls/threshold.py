import numpy as np
import sys
import os
from dotenv import load_dotenv
from scipy.stats import chi2

load_dotenv()

CONTROL_PATH = os.getenv("CONTROL_PATH")
sys.path.append(CONTROL_PATH)

from t2_control_limit import calculate_t_square
import matplotlib.pyplot as plt


def evaluate_threshold(data: np.array, alpha=0.05) -> np.array:

    t2_values = calculate_t_square(data, mode="single")
    t2_values = np.array(t2_values)

    num_variables = data.shape[1]
    threshold = run_bootstrap(t2_values, alpha)

    di_chart = []
    for column in range(num_variables):

        t_i = data[:, column]
        t_i2 = np.multiply(t_i, t_i)

        d_i = np.abs(np.subtract(t2_values, t_i2))
        di_chart.append(d_i)

    di_chart = np.array(di_chart).T

    compared = np.greater(di_chart, threshold)
    counted = np.count_nonzero(compared, axis=0)
    print(counted)
    x_label = ["x1", "x2", "x3", "x4", "x5"]
    plt.bar(x_label, counted)
    plt.xticks(x_label)
    plt.show()

    return "Completed"


def run_bootstrap(t2_values, alpha):

    num_t2_values = t2_values.shape[0]
    B = 500
    select_threshold_idx = int(num_t2_values * ((1 - alpha)))

    thresholds = []
    for idx in range(B):
        sorted_choice = np.sort(np.random.choice(t2_values, num_t2_values))
        thresholds.append(sorted_choice[select_threshold_idx])

    result = np.average(np.array(thresholds))
    return result
