import numpy as np
import pandas as pd
import collections


class DataGenarator(object):
    def __init__(self):
        self.normals = pd.DataFrame(self.generate_random_norm(mode="concat"))
        self.abnormals = pd.DataFrame(self.generate_abnormals()).T

    def build_dataset(self):
        normal_labels = pd.DataFrame([1]*500)
        normals = pd.concat([self.normals,normal_labels],axis=1)

        abnormal_labels = pd.DataFrame([-1]*100)
        abnormals = pd.concat([self.abnormals,abnormal_labels],axis=1)
        dataset = pd.concat([normals, abnormals], axis=0)
        dataset.columns = [['v1','v2','v3','v4','v5','label']]
        return dataset.sample(frac=1)

    def generate_random_norm(self, mode):
        if mode == "concat":
            v1 = np.random.normal(5, 10, 500).reshape(500, 1)
            v2 = np.random.normal(3, 1.2, 500).reshape(500, 1)
            v3 = np.random.normal(4, 3, 500).reshape(500, 1)
            v4 = np.random.normal(6, 1, 500).reshape(500, 1)
            v5 = np.random.normal(5.2, 20, 500).reshape(500, 1)

            result = np.concatenate([v1, v2, v3, v4, v5], axis=-1)
            return result
        elif mode == "each":
            v1 = np.random.normal(5, 10, 500)
            v2 = np.random.normal(3, 1.2, 500)
            v3 = np.random.normal(4, 3, 500)
            v4 = np.random.normal(6, 1, 500)
            v5 = np.random.normal(5.2, 20, 500)

            result = [v1, v2, v3, v4, v5]
            return result
        else:
            ValueError("concat or None")

    def generate_abnormals(self):

        normals = pd.DataFrame(self.generate_random_norm(mode="concat"))
        normal_samples = normals.sample(frac=1)[:-100]
        Q1 = normal_samples.quantile(0.25)
        Q3 = normal_samples.quantile(0.75)
        IQR = Q3 - Q1

        randoms = np.random.randn(5, 100)
        result = randoms * IQR.values.reshape(5, 1)

        return result


if __name__ == "__main__":

    dg = DataGenarator()
    print(dg.build_dataset())
