from modules.utils.tensorrt_model import TensorrtInference
import yaml
import cv2
import numpy as np

class FaceEmbedding(TensorrtInference):
    def __init__(self, config_fp) -> None:
        with open(config_fp) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.onnx = config["face_embedding"]["onnx"]
        self.engine = config["face_embedding"]["engine"]
        self.fp16_mode = config["face_embedding"]["fp16_mode"]
        self.output_shape = config["face_embedding"]["output_shape"]
        self.input_shape = config["face_embedding"]["input_shape"]

        super().__init__(self.onnx, self.engine, self.fp16_mode)
    
    def inference(self, img):
        pred = self(img).reshape(self.output_shape)
        return pred
        
    def pre_process(self, img):
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2,0,1))
        return img

    def __del__(self):
        self.destroy()
    

