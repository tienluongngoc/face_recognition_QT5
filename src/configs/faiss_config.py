from .base_config import BaseConfig

class FaissConfig(BaseConfig):
	def __init__(self, config_path) -> None:
		super(FaissConfig, self).__init__(config_path)
	
	@property
	def dim(self):
		return self.config["dim"]
	
	@property
	def device(self):
		return self.config["device"]
	
	@property
	def reload_all_db_delay(self):
		return self.config["reload_all_db_delay"]
	
	@property
	def retrain_delay(self):
		return self.config["retrain_delay"]
	
	@property
	def solve_event_delay(self):
		return self.config["solve_event_delay"]
	
	@property
	def threshold(self):
		return self.config["threshold"]
	
	@property
	def model_path(self):
		return self.config["model_path"]