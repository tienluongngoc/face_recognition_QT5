from .model_config import ModelConfig

class ArcFaceConfig(ModelConfig):
	def __init__(self, config_path) -> None:
		super(ArcFaceConfig, self).__init__(config_path)
	
	@property
	def threshold(self):
		return self.config["threshold"]
	
	@property
	def input_mean(self):
		return self.config["input_mean"]
	
	@property
	def input_std(self):
		return self.config["input_std"]
	
	@property
	def engine_default(self):
		return self.config["engine_default"]

	@property
	def output_shape(self):
		return self.config["output_shape"]