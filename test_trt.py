from src.tensorrt.inference_trt import YOLOV5TRT
from src.tensorrt.post import *
import numpy as np
import torch
import cv2
# from utils.augmentations import letterbox

stride = 32
pt = False
device = "cuda"
model = YOLOV5TRT("/data/tienln/workspace/mpi/face_recognition_pc_old/yolov5/yolov5s_face.engine", device=device, dnn=False, data=None, fp16=False)



img0 = cv2.imread("/data/tienln/workspace/mpi/face_recognition_pc_old/images/result.jpg")  # BGR
# img = letterbox(img0, [640, 640], stride=stride, auto=pt)[0]
img = cv2.resize(img0, (640,640))
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)

im = torch.from_numpy(img).to(device)
im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim

pred = model(im, augment=False, visualize=False)
pred = non_max_suppression_face(pred, 0.5, 0.5)
print(pred[0].shape)
