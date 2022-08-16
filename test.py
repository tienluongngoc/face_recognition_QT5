import sys
sys.path.append("./src")
import cv2
import time


from inferences.face_detection.yolov5 import YOLOV5
from configs.yolov5_config import Yolov5Config
config = Yolov5Config("configs/models/yolov5.yaml")
yolov5 = YOLOV5(config)
img0 = cv2.imread("cr7.jpg")  # BGR
img, img0=  yolov5.pre_process(img0)
st = time.time()
pred = yolov5.inference(img)
print(pred.shape)
print(time.time()-st)

pred = yolov5.post_process(img,img0,pred)

for i in range(len(pred)):
    cv2.putText(img0, f"{pred[i][14]}", (pred[i][0], pred[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
    cv2.rectangle(img0, (pred[i][0], pred[i][1]), (pred[i][2], pred[i][3]), (255,255,0), 1)

cv2.imwrite("test.jpg", img0)