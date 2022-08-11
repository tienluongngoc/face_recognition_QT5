from datetime import datetime
import yaml

class BaseConfig:
	def __init__(self, config_path) -> None:
		self.config = {}
		assert config_path.endswith(".yaml"), \
			f"[{datetime.now()}][{self.__class__.__name__}]: \
					Error config file path, config file must be YAML file."
		with open(config_path) as f:
			self.config = yaml.load(f, Loader=yaml.FullLoader)
