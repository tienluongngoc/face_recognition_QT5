from modules.utils.tensorrt_model import TensorrtInference
from modules.face_detection.face_detection import FaceDetection
from modules.utils.detection import img_vis

import cv2
import torch

face_det = FaceDetection("configs/config.yaml")
img = cv2.imread("sample.jpg")

img,orgimg =  face_det.pre_process(img)
pred = face_det.detect(img)
pred = face_det.post_process(pred)

img_vis(img,orgimg,pred)