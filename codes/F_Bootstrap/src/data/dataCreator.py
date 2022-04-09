import pandas as pd
import numpy as np
from numpy.random import gamma
from numpy.random import multivariate_normal
import sys
import os
from dotenv import load_dotenv

load_dotenv()

DATA_CONFIG_PATH = os.getenv("DATA_CONFIG_PATH")

sys.path.append(DATA_CONFIG_PATH)
from dataConfig import (
    LOG_NORMAL_DISTRIBUTION_VALUES,
    MV_GAMMA_DISTRIBUTION_VALUES,
    NUM_SAMPLE,
    T_VALUES,
anomaly
)

from t_distribution import multivariate_t


class DataCreator(object):
    def __init__(self):
        self.num_sample = NUM_SAMPLE

    def generate_gamma(self):
        shape_list = MV_GAMMA_DISTRIBUTION_VALUES["shape_list"]
        scale_list = MV_GAMMA_DISTRIBUTION_VALUES["scale_list"]

        data_multi_gamma = pd.DataFrame()
        for i, (shape, scale) in enumerate(zip(shape_list, scale_list)):
            data = gamma(shape=shape, scale=scale, size=self.num_sample)
            data_multi_gamma = pd.concat(
                [data_multi_gamma, pd.DataFrame(data, columns=[f"X{i + 1}"])], axis=1
            )

        return data_multi_gamma

    def generate_log_normal(self, mode=None):
        mean = LOG_NORMAL_DISTRIBUTION_VALUES["mean"]
        cov = LOG_NORMAL_DISTRIBUTION_VALUES["cov"]
        data_multi_normal = multivariate_normal(
            mean=mean, cov=cov, size=self.num_sample
        )
        if mode == "raw":
            return data_multi_normal
        else:
            pass
        data_multi_log_normal = np.exp(data_multi_normal)  # log(X) -> X
        return data_multi_log_normal

    def generate_t(self):
        df = T_VALUES["df"]
        mean = T_VALUES["mean"]

        rng = np.random.RandomState(4)
        tmp = rng.random((500, 5))
        shape = np.matmul(tmp.T, tmp)
        dist1 = multivariate_t(mean, shape, df=df, seed=2)

        return dist1.rvs(size=500)

    # It is the concatenation of normal distribution and anomaly generated to study problem1
    def generate_problem1_data(self):
        normal = self.generate_log_normal(mode="raw")
        result = np.concatenate((normal,anomaly),axis=0)
        return result



if __name__ == "__main__":
    dc = DataCreator()
    print(dc.generate_log_normal())
    print(dc.generate_problem1_data().shape)
