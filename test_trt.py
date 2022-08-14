from distutils.command.config import config
import sys
sys.path.append("./src")
from trt_model import TRTModel, YOLOV5
from src.trt_model.yolov5_utils import *
import numpy as np
import torch
import cv2
# from utils.augmentations import letterbox

stride = 32
pt = False
device = "cuda"
config = {"weights": "/data/tienln/workspace/mpi/face_recognition_pc_old/yolov5/yolov5s_face.engine",
          "device": "cuda",
          "fp16": False,
          "input_name": "input",
          "output_name": "output"}
model = YOLOV5(config)

img0 = cv2.imread("/data/tienln/workspace/mpi/face_recognition_pc_old/images/result.jpg")  # BGR
img=  model.pre_process(img0)
# print(img.shape)
pred = model.inference(img)
pred = model.post_process(img,img0,pred)
for i in range(len(pred)):
    cv2.putText(img0, f"{pred[i][14]}", (pred[i][0], pred[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
    cv2.rectangle(img0, (pred[i][0], pred[i][1]), (pred[i][2], pred[i][3]), (255,255,0), 1)

cv2.imwrite("test.jpg", img0)
# print(pred)
# print(pred)


pass
img = letterbox(img0, [640, 640], stride=stride, auto=pt)[0]
print(img.shape)
# img = cv2.resize(img0, (640,640))
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)

im = torch.from_numpy(img).to(device)
im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim
print(im.shape)
pred = model.inference(im)
pred = non_max_suppression_face(pred, 0.5, 0.5)
# print(pred[0].shape)
