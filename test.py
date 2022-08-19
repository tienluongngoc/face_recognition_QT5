import sys
sys.path.append("./src")
import cv2
import time


from inferences.face_detection.yolov5 import YOLOV5
from configs.yolov5_config import Yolov5Config
from configs.arcface_trt_config import ArcFaceTRTConfig
from inferences.face_encode.arcface_trt import ArceFaceTRT
# config = Yolov5Config("configs/models/yolov5.yaml")
# yolov5 = YOLOV5(config)
# img0 = cv2.imread("cr7.jpg")  # BGR
# pred = yolov5.detect(img0)
# print(pred[0][0][4])
# print(pred[1])
# for i in range(len(pred)):
#     cv2.putText(img0, f"{pred[i][14]}", (pred[i][0], pred[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
#     cv2.rectangle(img0, (pred[i][0], pred[i][1]), (pred[i][2], pred[i][3]), (255,255,0), 1)

# cv2.imwrite("test.jpg", img0)


config = ArcFaceTRTConfig("configs/models/arcface_trt.yaml")
arcface = ArceFaceTRT(config)
img = cv2.imread("cr7.jpg")  # BGR
arcface.get_feat(img)

