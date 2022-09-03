from src.inferences.face_detection.yolov5_torch import YOLOV5Torch
from src.inferences.face_detection.yolov5_models.detect_face import detect_one
from src.configs.yolov5_torch_config import Yolov5TorchConfig
import cv2
img= cv2.imread("tienln.jpg")
config = Yolov5TorchConfig("configs\models\yolov5_torch.yaml")
det = YOLOV5Torch(config)
det.detect(img)