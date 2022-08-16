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

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]