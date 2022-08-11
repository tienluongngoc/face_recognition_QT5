from .model_config import ModelConfig

class SCRFDConfig(ModelConfig):
	def __init__(self, config_path) -> None:
		super(SCRFDConfig, self).__init__(config_path)
	
	@property
	def fmc(self):
		return self.config["fmc"]
	
	@property
	def feat_stride_fpn(self):
		return self.config["feat_stride_fpn"]
	
	@property
	def num_anchors(self):
		return self.config["num_anchors"]
	
	@property
	def input_mean(self):
		return self.config["input_mean"]
	
	@property
	def input_std(self):
		return self.config["input_std"]
	
	@property
	def det_thresh(self):
		return self.config["det_thresh"]
	
	@property
	def nms_thresh(self):
		return self.config["nms_thresh"]