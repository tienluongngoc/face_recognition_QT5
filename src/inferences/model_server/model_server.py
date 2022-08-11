import tritonclient.grpc as grpcclient
import requests
from datetime import datetime
from ..base import Singleton

class TritonModelServer(metaclass=Singleton):
	def __init__(self, config) -> None:
		triton_ip = config["host"]
		triton_port = config["port"]
		url_check = f"{triton_ip}:{triton_port - 1}"
		try:
			res = requests.get("http://" + url_check + "/v2/health/ready")
			assert res.ok and res.status_code == 200, f"[{datetime.now()}][{self.__class__.__name__}]: Failed to connect to model server"    
		except requests.exceptions.RequestException as e:
			print(f"[{datetime.now()}][{self.__class__.__name__}]: {e}")
			
		url = f"{triton_ip}:{triton_port}"
		self.triton_client = grpcclient.InferenceServerClient(url=url)

	def initialize(self, model_name, input_names, input_size, output_names) -> None:
		if model_name == "yolov5_idcard_back":
			self.initialize_yolov5_idcard_back(model_name, input_names, input_size, output_names)
		elif model_name == "yolov5_idcard_front":
			self.initialize_yolov5_idcard_front(model_name, input_names, input_size, output_names)
		elif model_name == "scrfd":
			self.initialze_face_detection(model_name, input_names, input_size, output_names)
		elif model_name == "arcface":
			self.initialize_face_encode(model_name, input_names, input_size, output_names)
		elif model_name == "fasnetv1se" or model_name == "fasnetv2":
			self.initialize_face_anti_spoofing(model_name, input_names, input_size, output_names)
			

	def infer(self, model_name, input) -> grpcclient.InferResult:
		if model_name == "yolov5_idcard_back":
			return self.get_yolov5_idcard_back(input)
		elif model_name == "yolov5_idcard_front":
			return self.get_yolov5_idcard_front(input)
		elif model_name == "scrfd":
			return self.get_face_detection(input)
		elif model_name == "arcface":
			return self.get_face_encode(input)
		elif model_name == "fasnetv1se" or model_name == "fasnetv2":
			return self.get_face_anti_spoofing(input)


	def initialze_face_detection(self, model_name, input_names, input_size, output_names) -> None:
		self.fd_model_name = model_name
		assert self.triton_client.is_model_ready(model_name), f"[{datetime.now()}][{self.__class__.__name__}]: Model {model_name} is not ready"
		self.grpc_fd_inputs = [grpcclient.InferInput(name, [1, 3, *input_size], "FP32") for name in input_names]
		self.grpc_fd_outputs = [grpcclient.InferRequestedOutput(output) for output in output_names]

	def initialize_face_encode(self, model_name, input_names, input_size, output_names) -> None:
		self.fe_model_name = model_name
		assert self.triton_client.is_model_ready(model_name), f"[{datetime.now()}][{self.__class__.__name__}]: Model {model_name} is not ready"
		self.grpc_fe_inputs = [grpcclient.InferInput(name, [1, 3, *input_size], "FP32") for name in input_names]
		self.grpc_fe_outputs = [grpcclient.InferRequestedOutput(output) for output in output_names]

	def initialize_face_anti_spoofing(self, model_name, input_names, input_size, output_names) -> None:
		self.fas_model_name = model_name
		assert self.triton_client.is_model_ready(model_name), f"[{datetime.now()}][{self.__class__.__name__}]: Model {model_name} is not ready"
		self.grpc_fas_inputs = [grpcclient.InferInput(name, [1, 3, *input_size], "FP32") for name in input_names]
		self.grpc_fas_outputs = []

	def initialize_yolov5_idcard_front(self, model_name, input_names, input_size, output_names) -> None:
		self.idc_fr_model_name = model_name
		assert self.triton_client.is_model_ready(model_name), f"[{datetime.now()}][{self.__class__.__name__}]: Model {model_name} is not ready"
		self.grpc_fr_inputs = [grpcclient.InferInput(name, [1, 3, *input_size], "FP32") for name in input_names]
		self.grpc_fr_outputs = [grpcclient.InferRequestedOutput(output) for output in output_names]

	def initialize_yolov5_idcard_back(self, model_name, input_names, input_size, output_names) -> None:
		self.idc_bk_model_name = model_name
		assert self.triton_client.is_model_ready(model_name), f"[{datetime.now()}][{self.__class__.__name__}]: Model {model_name} is not ready"
		self.grpc_bk_inputs = [grpcclient.InferInput(name, [1, 3, *input_size], "FP32") for name in input_names]
		self.grpc_bk_outputs = [grpcclient.InferRequestedOutput(output) for output in output_names]


	def get_face_detection(self, input) -> grpcclient.InferResult:
		self.grpc_fd_inputs[0].set_data_from_numpy(input)
		results = self.triton_client.infer(
			model_name=self.fd_model_name,
			inputs=self.grpc_fd_inputs,
			outputs=self.grpc_fd_outputs,
			headers={}
		)
		return results

	def get_face_encode(self, input) -> grpcclient.InferResult:
		self.grpc_fe_inputs[0].set_data_from_numpy(input)
		results = self.triton_client.infer(
			model_name=self.fe_model_name,
			inputs=self.grpc_fe_inputs,
			outputs=self.grpc_fe_outputs,
			headers={}
		)
		return results
	
	def get_face_anti_spoofing(self, input) -> grpcclient.InferResult:
		self.grpc_fas_inputs[0].set_data_from_numpy(input)
		results = self.triton_client.infer(
			model_name=self.fas_model_name,
			inputs=self.grpc_fas_inputs,
			outputs=self.grpc_fas_outputs,
			headers={}
		)
		return results
	
	def get_yolov5_idcard_front(self, input) -> grpcclient.InferResult:
		self.grpc_fr_inputs[0].set_data_from_numpy(input)
		results = self.triton_client.infer(
			model_name=self.idc_fr_model_name,
			inputs=self.grpc_fr_inputs,
			outputs=self.grpc_fr_outputs,
			headers={}
		)
		return results

	def get_yolov5_idcard_back(self, input) -> grpcclient.InferResult:
		self.grpc_bk_inputs[0].set_data_from_numpy(input)
		results = self.triton_client.infer(
			model_name=self.idc_bk_model_name,
			inputs=self.grpc_bk_inputs,
			outputs=self.grpc_bk_outputs,
			headers={}
		)
		return results