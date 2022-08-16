from .model_config import ModelConfig

class Yolov5Config(ModelConfig):
	def __init__(self, config_path) -> None:
		super(Yolov5Config, self).__init__(config_path)
	
	@property
	def onnx(self):
		result = self.config["onnx"]
		return result

	@property
	def engine(self):
		result = self.config["engine"]
		return result

	@property
	def device(self):
		result = self.config["device"]
		return result

	@property
	def half(self):
		result = self.config["half"]
		return result

	@property
	def input_name(self):
		result = self.config["input_name"]
		return result
	
	@property
	def input_shape(self):
		result = self.config["input_shape"]
		return result

	@property
	def output_name(self):
		result = self.config["output_name"]
		return result

	@property
	def output_shape(self):
		result = self.config["output_shape"]
		return result

	@property
	def max_workspace(self):
		result = self.config["max_workspace"]
		return result


	@property
	def stride_max(self):
		result = self.config["stride_max"]
		return result

	@property
	def confidence_threshold(self):
		result = self.config["confidence_threshold"]
		return result

	@property
	def iou_threshold(self):
		result = self.config["iou_threshold"]
		return result