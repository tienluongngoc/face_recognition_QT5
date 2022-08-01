# from utils.logger import Logger
from modules.face_detection.retina_face import YOLOV5TRT
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device, time_sync
import cv2
import torch
import numpy as np

model = YOLOV5TRT()
device = select_device("0")
print()
dataset = LoadImages("t1.jpg", img_size=[640, 640], stride=32, auto=False)
for path, im, im0s, vid_cap, s in dataset:
    # t1 = time_sync()
    # print(im.shape)
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # print("check: ", im.shape)
    # t2 = time_sync()
    # dt[0] += t2 - t1
    
    # Inference
    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    print(im.shape)
    pred = model(im, augment=False, visualize=False)
    # print(pred.shape)
    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
       

# img = cv2.imread('t1.jpg')
# img = cv2.resize(img, (640, 640))
# img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
# img = np.ascontiguousarray(img)
# im = torch.from_numpy(img).to("cuda")
# if len(im.shape) == 3:
#     im = im[None]  # expand for batch dim
# print(im.shape)
# face(im)