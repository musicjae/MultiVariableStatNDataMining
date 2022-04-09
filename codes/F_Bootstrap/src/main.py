import numpy as np
np.random.seed(seed=827)
from data.dataCreator import DataCreator
import sys
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
SRC_PATH = os.getenv("SRC_PATH")
sys.path.append(SRC_PATH)
from controls import t2_control_limit as t2
from controls import average, threshold, rank_sum
from parameter import ALPHA, args


def problem1(mode=None):
    data_creator = DataCreator()
    data = data_creator.generate_problem1_data()

    if args.select_method == "average":
        result = average.evaluate_average(data)
    elif args.select_method == "ranksum":
        result = rank_sum.evaluate_ranksum(data)
    elif args.select_method == "threshold":
        result = threshold.evaluate_threshold(data)

    return result


def problem2(mode=None):
    data_creator = DataCreator()

    gamma_dist = data_creator.generate_gamma()
    log_normal_dist = data_creator.generate_log_normal()
    t_dist = data_creator.generate_t()

    if mode == "gamma":
        for a in ALPHA:
            t2.control_limit(gamma_dist, alpha=a, data_dist_name="gamma")
    elif mode == "log_normal":
        for a in ALPHA:
            t2.control_limit(log_normal_dist, alpha=a, data_dist_name="log_normal")
    elif mode == "t":
        for a in ALPHA:
            t2.control_limit(t_dist, alpha=a, data_dist_name="t")
    else:
        return "Enter at least one of gamma, log_normal, t"


if __name__ == "__main__":
    if args.problem == 1:
        print(problem1())
    else: # problem 2
        if args.select_dist == "gamma":
            problem2(mode="gamma")
        elif args.select_dist == "log_normal":
            problem2(mode="log_normal")
        elif args.select_dist == "t":
            problem2(mode="t")
        else:
            print("다시 입력하세요")

