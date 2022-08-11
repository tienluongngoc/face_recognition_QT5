from .model_config import ModelConfig

class FASNetConfig(ModelConfig):
	def __init__(self, config_path) -> None:
		super(FASNetConfig, self).__init__(config_path)
	
	@property
	def scale(self):
		return self.config["scale"]
	
	@property
	def threshold(self):
		return self.config["threshold"]