import numpy as np
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
def test_hypothesis(data):

    num_sample = data.shape[0]
    num_variables = data.shape[1]

    normal_data = data[:500,:]
    abnormal_data = data[500:,:]

    normal_avgs=[]
    sampled_avgs=[]
    sampleds=[]
    for column in range(num_variables):

        normal = normal_data[:,column]
        normal_average = np.average(normal,axis=0)
        normal_avgs.append(normal_average)

        sampled_normal = np.random.choice(normal_data[:,column],10)
        sampled_normal_average = np.average(sampled_normal,axis=0)
        sampled_avgs.append(sampled_normal_average)
        sampleds.append(sampled_normal)

    sampleds = np.array(sampleds).T
    normal_avgs = np.array(normal_avgs)
    sampled_avgs = np.array(sampled_avgs)

    denominator = np.abs(sampled_avgs - normal_avgs)
    numerator = np.std(sampleds) / np.sqrt(num_sample)

    tau = np.divide(denominator,numerator)

    abnormal_avgs = np.average(abnormal_data,axis=0)
    print(tau,"tau")
    diff = np.abs(np.subtract(sampled_avgs,abnormal_avgs))

    result = np.abs(np.subtract(tau,diff))

    result_per = np.divide(result,np.sum(result))
    print(result_per,"result_percent")
    x_label = ["x1", "x2", "x3", "x4", "x5"]
    plt.bar(x_label, result_per)
    plt.xticks(x_label)
    plt.show()

    return result_per

