from .base_config import BaseConfig
from .scrfd_config import SCRFDConfig
from .arcface_config import ArcFaceConfig
from .faiss_config import FaissConfig
from .fasnet_config import FASNetConfig

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
	def detection(self):
		return SCRFDConfig(self.config["models"]["detection_path"])
	
	@property
	def encode(self):
		return ArcFaceConfig(self.config["models"]["encode_path"])

	@property
	def anti_spoofing_v1se(self):
		return FASNetConfig(self.config["models"]["face_anti_spoofing_path"]["v1se"])
	
	@property
	def anti_spoofing_v2(self):
		return FASNetConfig(self.config["models"]["face_anti_spoofing_path"]["v2"])

	@property
	def recognition(self):
		return FaissConfig(self.config["models"]["recognition_path"])
	
	@property
	def api(self):
		return self.config["api"]
	
	@property
	def mongodb(self):
		return self.config["mongodb"]
	
	@property
	def faces(self):
		return self.config["faces"]
	
	@property
	def model_server(self):
		return self.config["model_server"]
	