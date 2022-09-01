# from distutils.command.config import config
# from statistics import mode
# import sys
# sys.path.append("./src")
# from trt_model import TRTModel, YOLOV5
# from src.trt_model.yolov5_utils import *
# import numpy as np
# import torch
# import cv2
# # from utils.augmentations import letterbox
# import time
# stride = 32
# pt = False
# device = "cuda"
# config = {"onnx": "weights/yolov5s_face.onnx", 
#           "engine": "weights/yolov5s_face_fp16.engine",
#           "device": "cuda",
#           "half": True,
#           "max_workspace_size": 4,
#           "input_name": "input",
#           "output_name": "output"}
# model = YOLOV5(config)
# # model.export_engine()

# img0 = cv2.imread("/data/tienln/workspace/mpi/face_recognition_pc_old/images/result.jpg")  # BGR
# img, img0=  model.pre_process(img0)
# print(img.shape)
# for i in range(10):
#     st = time.time()
#     pred = model.inference(img)
#     print(time.time()-st)
# pred = model.post_process(img,img0,pred)
# for i in range(len(pred)):
#     cv2.putText(img0, f"{pred[i][14]}", (pred[i][0], pred[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
#     cv2.rectangle(img0, (pred[i][0], pred[i][1]), (pred[i][2], pred[i][3]), (255,255,0), 1)

# cv2.imwrite("test.jpg", img0)

import sys
sys.path.append("./src")

from configs.arcface_trt_config import ArcFaceTRTConfig
from inferences.face_encode.arcface_trt import ArceFaceTRT
import cv2
from configs.yolov5_config import Yolov5Config
from inferences.face_detection.yolov5 import YOLOV5

# config = ArcFaceTRTConfig("configs/models/arcface_trt.yaml")
# arcface = ArceFaceTRT(config)

img = cv2.imread("cr7.png")
config = Yolov5Config("configs/models/yolov5.yaml")
yolo5 = YOLOV5(config)

yolo5.detect(img)