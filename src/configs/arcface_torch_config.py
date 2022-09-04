from .model_config import ModelConfig

class ArcFaceTorchConfig(ModelConfig):
	def __init__(self, config_path) -> None:
		super(ArcFaceTorchConfig, self).__init__(config_path)
	
	@property
	def weight(self):
		return self.config["weight"]

	@property
	def device(self):
		return self.config["device"]

	
	@property
	def backbone(self):
		return self.config["backbone"]
	
	@property
	def engine_default(self):
		return self.config["engine_default"]

	