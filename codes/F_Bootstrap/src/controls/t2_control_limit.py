import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt


def control_limit(data, alpha: float, data_dist_name: str, bs_size: int = 100):

    t2_values, m, p = calculate_t_square(data)

    # control limit with f distribution
    CL = (p * (m + 1) * (m - 1)) / (m * (m - p))
    UCL_f = CL * f.ppf(1 - alpha, p, m - p)

    # control limit with bootstrap
    quantile_list = []
    for i in range(bs_size):
        samples = np.random.choice(t2_values, size=len(t2_values), replace=True)
        quantile = np.quantile(samples, 1 - alpha)
        quantile_list.append(quantile)
    UCL_bs = np.mean(quantile_list)

    # plot control chart
    plt.figure(figsize=(10, 5))
    plt.plot(t2_values, color="b")
    plt.axhline(UCL_f, color="r", label="F distribution", linestyle="--")
    plt.axhline(UCL_bs, color="g", label="Bootstrap", linestyle="--")
    plt.legend()
    plt.title(
        f"T2 Control Chart(F vs Bootstrap), {data_dist_name}, alpha = {alpha}",
        fontweight="bold",
    )
    plt.ylim((0, 100))
    plt.xlabel("Observation")
    plt.ylabel("T2")
    plt.show()

    print("-" * 10 + " False alarm with F distribution " + "-" * 10)
    print(f"Expected false alarm rate: {alpha}")
    print(f"Data false alarm rate: {(t2_values > UCL_f).sum() / len(data)}")

    print("-" * 10 + " False alarm with Bootstrap " + "-" * 10)
    print(f"Expected false alarm rate: {alpha}")

    return "Completed"


def calculate_t_square(data, mode="multiple"):
    array_data = np.array(data)
    m = len(data)  # number of samples
    p = array_data.shape[1]  # number of variables

    x_mean = np.mean(array_data, axis=0)  # mean of variables
    cov = np.cov(array_data.T)  # covariance
    cov_inv = np.linalg.inv(cov)  # inverse S

    t2_values = []
    for sample in array_data:
        dif = sample - x_mean
        t2 = (dif.T).dot(cov_inv).dot(dif)  # T-square
        t2_values.append(t2)

    if mode == "single":
        return t2_values
    else:
        return t2_values, m, p
