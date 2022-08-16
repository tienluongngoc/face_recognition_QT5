import sys
sys.path.append("./src")

from inferences.face_detection.yolov5 import YOLOV5
from configs.yolov5_config import Yolov5Config
config = Yolov5Config("configs/models/yolov5.yaml")
yolov5 = YOLOV5(config)