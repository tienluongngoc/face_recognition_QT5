# # import sys
# # sys.path.append("./src")

# from src.configs.arcface_trt_config import ArcFaceTRTConfig
# from src.inferences.face_encode.arcface_trt import ArceFaceTRT
# import cv2
# from src.configs.yolov5_config import Yolov5Config
# from src.inferences.face_detection.yolov5 import YOLOV5

# config = ArcFaceTRTConfig("configs/models/arcface_trt.yaml")
# arcface = ArceFaceTRT(config)
# img = cv2.imread("cr7.png")
# config = Yolov5Config("configs/models/yolov5.yaml")
# yolo5 = YOLOV5(config)

# yolo5.detect(img)


import code
from distutils.command.config import config
from src.inferences.face_detection.scrfd import SCRFD
from src.configs.scrfd_config import SCRFDConfig
from src.inferences.face_encode.arcface import ArcFace
from src.configs.arcface_config import ArcFaceConfig
from src.inferences.face_encode.arcface_ultils.models import get_model
from src.inferences.face_encode.arcface_torch import  ArcFaceTorch
import cv2
import time
# config = SCRFDConfig("configs\models\scrfd.yaml")
# detection = SCRFD(config)
img = cv2.imread("images\download.jpg")
# det, kpss = detection.detect(img)
# # print(img)
# for i in range(1000):
#     st = time.time()
#     det, kpss = detection.detect(img)
#     print((time.time()-st)*1000)


config = ArcFaceConfig("configs\models\\arcface.yaml")
# encode = ArcFace(config)
arcface_torch = ArcFaceTorch(config)
# for i in range(1000):
#     st = time.time()
# code = encode.get_feat(img)    
    # print((time.time()-st)*1000)
# code = arcface_torch.get_feet(img)
# print(type(code))
# net = get_model("r100", fp16=False)