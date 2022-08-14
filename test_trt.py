from distutils.command.config import config
from statistics import mode
import sys
sys.path.append("./src")
from trt_model import TRTModel, YOLOV5
from src.trt_model.yolov5_utils import *
import numpy as np
import torch
import cv2
# from utils.augmentations import letterbox
import time
stride = 32
pt = False
device = "cuda"
config = {"onnx": "/data/tienln/workspace/mpi/face_recognition_pc_old/models/yolov5s_face.onnx", 
          "engine": "/data/tienln/workspace/mpi/face_recognition_pc_old/yolov5/yolov5s_face.engine",
          "device": "cuda",
          "fp16": False,
          "input_name": "input",
          "output_name": "output"}
model = YOLOV5(config)
# model.export_engine()

img0 = cv2.imread("/data/tienln/workspace/mpi/face_recognition_pc_old/images/result.jpg")  # BGR
img, img0=  model.pre_process(img0)
# print(img.shape)
for i in range(10):
    st = time.time()
    pred = model.inference(img)
    print(time.time()-st)
pred = model.post_process(img,img0,pred)
for i in range(len(pred)):
    cv2.putText(img0, f"{pred[i][14]}", (pred[i][0], pred[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
    cv2.rectangle(img0, (pred[i][0], pred[i][1]), (pred[i][2], pred[i][3]), (255,255,0), 1)

cv2.imwrite("test.jpg", img0)
