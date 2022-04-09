import numpy as np

NUM_SAMPLE = 500

MV_GAMMA_DISTRIBUTION_VALUES = {
    "shape_list": [0.1, 0.1, 0.1, 0.1, 0.1],
    "scale_list": [2, 0.7, 1, 0.7, 1],
}

LOG_NORMAL_DISTRIBUTION_VALUES = {
    "mean": [0.0, 2.0, 3.0, 4.0, 5.0],
    "cov": [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ],
}
T_VALUES = {"df": 1, "loc": [1.0, 2.0, 3.0, 4.0, 5.0], "mean": [0, 0, 0, 0, 0]}

anomaly = np.random.rand(50).reshape(10,5) * 2
