from .fasnet import MiniFASNet
from configs import FASNetConfig
from typing import List
import numpy as np
from ..base import Singleton

class MiniFASNetEmsemble(metaclass=Singleton):
	def __init__(self, configs: List[FASNetConfig]) -> None:
		self.mini_fasnet_v1se = MiniFASNet(configs[0])
		self.mini_fasnet_v2 = MiniFASNet(configs[1])
		self.threshold = (configs[0].threshold + configs[1].threshold) / 2
	
	def check(self, image: np.ndarray, bbox: np.ndarray):
		img_crop1 = self.mini_fasnet_v1se.align_bbox(image, bbox)
		img_crop2 = self.mini_fasnet_v2.align_bbox(image, bbox)
		result1 = self.mini_fasnet_v1se.forward(img_crop1)
		result2 = self.mini_fasnet_v2.forward(img_crop2)
		results = (result1 + result2) / 2
		label = np.argmax(results)
		score = results[label]
		if label == 1 and score >= self.threshold:
			return True
		return False