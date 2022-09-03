from .model_config import ModelConfig

class Yolov5TorchConfig(ModelConfig):
	def __init__(self, config_path) -> None:
		super(Yolov5TorchConfig, self).__init__(config_path)
	
	@property
	def weight(self):
		result = self.config["weight"]
		return result

	@property
	def device(self):
		result = self.config["device"]
		return result

	@property
	def input_shape(self):
		result = self.config["input_shape"]
		return result

	@property
	def confidence_threshold(self):
		result = self.config["confidence_threshold"]
		return result

	@property
	def iou_threshold(self):
		result = self.config["iou_threshold"]
		return result