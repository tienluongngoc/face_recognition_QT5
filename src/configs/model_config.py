from .base_config import BaseConfig

class ModelConfig(BaseConfig):
	def __init__(self, config_path) -> None:
		super(ModelConfig, self).__init__(config_path)
	
	@property
	def model_name(self):
		return self.config["model_name"]
	
	@property
	def remote(self):
		return self.config["remote"]

	@property
	def model_path(self):
		return self.config["model_path"]
	
	@property
	def device(self):
		return self.config["device"]
	
	@property
	def input_width(self):
		return self.config["input_width"]
	
	@property
	def input_height(self):
		return self.config["input_height"]
	
	@property
	def input_names(self):
		return self.config["input_names"]
	
	@property
	def output_names(self):
		return self.config["output_names"]
