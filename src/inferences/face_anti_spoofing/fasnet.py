from __future__ import division
import numpy as np
from datetime import datetime
from src.configs.fasnet_config import FASNetConfig
from ..utils.face_anti import CropImage, softmax
from uvicorn.config import logger

class MiniFASNet():
	def __init__(self, config: FASNetConfig) -> None:
		self.config = config
		MiniFASNet.__name__ = self.config.model_name
		self.input_size = tuple([config.input_width, config.input_height])
		if config.remote:
			from ..model_server import Triton_inference
			Triton_inference.initialize(
				model_name=config.model_name, 
				input_names=config.input_names, 
				input_size=self.input_size,
				output_names=config.output_names	
			)
			logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Using anti spoofing model from model server.")
		else:
			import onnxruntime
			if config.device == "gpu":
				provider = "CUDAExecutionProvider"
			elif config.device == "cpu":
				provider = "CPUExecutionProvider"
			elif config.device == "tensorrt":
				provider = "TensorrtExecutionProvider"
			else:
				assert False, f"[{datetime.now()}][{self.__class__.__name__}]: Error device, device is only one \
					of three values: ['cpu', 'gpu', 'tensorrt']"
			self.session = onnxruntime.InferenceSession(config.model_path, providers=[provider])
			logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Using local anti spoofing model.")

		self.cropper = CropImage()

	def forward(self, image: np.ndarray):
		image_cp = image.copy()
		image_cp = np.transpose(image_cp, (2, 0, 1))
		image_cp = np.expand_dims(image_cp, 0)
		if self.config.remote:
			from ..model_server import Triton_inference
			results = Triton_inference.infer(self.config.model_name, image_cp)
			net_outs = [results.as_numpy(name) for name in self.config.output_names]
		else:
			net_outs = self.session.run(self.config.output_names, {self.config.input_names[0] : image_cp})
		net_outs = softmax(net_outs[0][0])
		# logger.info(net_outs)
		return net_outs

	def align_bbox(self, image: np.ndarray, bbox):
		image_cp = image.copy()
		param = {
			"org_img": image_cp,
			"bbox": bbox,
			"scale": self.config.scale,
			"out_w": self.config.input_width,
			"out_h": self.config.input_height,
			"crop": True,
		}
		crop_img = self.cropper.crop(**param)
		return np.array(crop_img, dtype=np.float32)
	
	def check(self, image: np.ndarray, bbox: np.ndarray):
		crop_img = self.align_bbox(image, bbox)
		net_outs = self.forward(crop_img)
		label = np.argmax(net_outs)
		score = net_outs[label]
		if label == 1 and score >= self.config.threshold:
			return True
		return False