import argparse

parser = argparse.ArgumentParser(description='parameters.....')
parser.add_argument('--select_dist', type=str,default='gamma')
args = parser.parse_args()

ALPHA = [0.01, 0.05, 0.1, 0.2]
DATA_DIST_NAME = {"Gamma": "Gamma", "Log-normal": "Log-normal","t":"t"}
