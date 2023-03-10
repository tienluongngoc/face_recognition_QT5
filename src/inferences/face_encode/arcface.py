from src.configs.arcface_config import ArcFaceConfig
from ..utils import face_align, face_detect
from datetime import datetime
from ..base import Singleton
import numpy as np
import onnxruntime
import cv2


class ArcFace(metaclass=Singleton):
	def __init__(self, config: ArcFaceConfig) -> None:
		self.config = config
		self.input_size = tuple([config.input_width, config.input_height])
		self.output_shape = tuple([1, config.output_shape])

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
	
	def get(self, img: np.ndarray, face: face_detect.Face):
		aimg = face_align.norm_crop(img, landmark=face.kps)
		face.embedding = self.get_feat(aimg).flatten()
		return face.normed_embedding
	
	def get_feat(self, imgs: np.ndarray):
		if not isinstance(imgs, list):
			imgs = [imgs]
		input_size = self.input_size
		
		blob = cv2.dnn.blobFromImages(
			imgs, 1.0 / self.config.input_std, input_size,
			(self.config.input_mean, self.config.input_mean, self.config.input_mean), swapRB=True
		)
		
		net_out = self.session.run(self.config.output_names, {self.config.input_names[0]: blob})[0]
		return net_out
	
	def compute_sim(self, feat1, feat2):
		from numpy.linalg import norm
		feat1 = feat1.ravel()
		feat2 = feat2.ravel()
		sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
		return sim

	def is_same(self, feat1, feat2):
		sim = self.compute_sim(feat1, feat2)
		if sim > self.config.threshold:
			return True
		else:
			return False