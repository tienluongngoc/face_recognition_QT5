import yaml
import numpy as np

def load_config(path):
	with open(path, 'r') as config_file:
		config = yaml.load(config_file, Loader=yaml.CLoader)	
	return config

def npfloat2float(arr: np.ndarray):
	list_arr = list(arr)
	list_arr = [float(x) for x in arr]
	return list_arr