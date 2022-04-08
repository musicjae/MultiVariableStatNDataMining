from data.dataCreator import DataCreator
import sys

sys.path.append(
    "/Users/jeongjaeyeong/Desktop/School/2022-1/다변량통계 및 데이터 마이닝/HW/dev/MultiVariableStatNDataMining/codes/hw3_F_Bootstrap/src"
)
from controls import t2_control_limit as t2
from parameter import ALPHA,args


def main(mode=None):
    data_creator = DataCreator()

    gamma_dist = data_creator.generate_gamma()
    log_normal_dist = data_creator.generate_log_normal()
    t_dist = data_creator.generate_t()

    if mode == "gamma":
        for a in ALPHA:
            t2.control_limit(gamma_dist,alpha=a,data_dist_name='gamma')
    elif mode == "log_normal":
        for a in ALPHA:
            t2.control_limit(log_normal_dist,alpha=a,data_dist_name='log_normal')
    elif mode == "t":
        for a in ALPHA:
            t2.control_limit(t_dist,alpha=a,data_dist_name="t")
    else:
        return "Enter at least one of gamma, log_normal, t"


if __name__ == "__main__":
    if args.select_dist == "gamma":
        main(mode="gamma")
    elif args.select_dist == "log_normal":
        main(mode="log_normal")
    elif args.select_dist == "t":
        main(mode="t")
    else:
        print("다시 입력하세요")