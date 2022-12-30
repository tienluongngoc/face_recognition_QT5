from .arcface_trt_config import ArcFaceTRTConfig
from .base_config import BaseConfig
from .scrfd_config import SCRFDConfig
from .arcface_config import ArcFaceConfig
from .faiss_config import FaissConfig
from .yolov5_config import Yolov5Config
from .ui_config import UIConfig

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FaceRecogAPIConfig(BaseConfig, metaclass=Singleton):
	def __init__(self, config_path) -> None:
		super(FaceRecogAPIConfig, self).__init__(config_path)
		
	@property
	def detection_engine(self):
		engine_name = self.config["models"]["detection"]["engine"]
		return engine_name
	
	@property
	def detection(self):
		if self.config["models"]["detection"]["engine"] == "scrfd":
			result = SCRFDConfig(self.config["models"]["detection"]["engine_config"]["scrfd"])
		elif self.config["models"]["detection"]["engine"] == "yolov5":
			result = Yolov5Config(self.config["models"]["detection"]["engine_config"]["yolov5"])
		elif self.config["models"]["detection"]["engine"] == "yolov5_torch":
			result = Yolov5TorchConfig(self.config["models"]["detection"]["engine_config"]["yolov5_torch"])
		else:
			pass
		return result
	
	@property
	def encode(self):
		if self.config["models"]["encode"]["engine"] == "arcface":
			result = ArcFaceConfig(self.config["models"]["encode"]["engine_config"]["arcface"])
		elif self.config["models"]["encode"]["engine"] == "arcface_trt":
			result = ArcFaceTRTConfig(self.config["models"]["encode"]["engine_config"]["arcface_trt"])
		elif self.config["models"]["encode"]["engine"] == "arcface_torch":
			result = ArcFaceTorchConfig(self.config["models"]["encode"]["engine_config"]["arcface_torch"])
		else:
			pass
		return result

	@property
	def encode_engine(self):
		engine_name = self.config["models"]["encode"]["engine"]
		return engine_name

	@property
	def recognition(self):
		if self.config["models"]["recognition"]["engine"] == "faiss_cpu":
			result = FaissConfig(self.config["models"]["recognition"]["engine_config"]["faiss_cpu"])
		else:
			pass
		return result

	@property
	def recognition_engine(self):
		engine_name = self.config["models"]["recognition"]["engine"]
		return engine_name
	
	@property
	def mongodb(self):
		return self.config["mongodb"]
	
	@property
	def faces(self):
		return self.config["faces"]
	
