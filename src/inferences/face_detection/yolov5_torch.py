import torch.backends.cudnn as cudnn
from pathlib import Path
from numpy import random
import numpy as np
import argparse
import torch
import time
import cv2
import copy

from ...configs.yolov5_torch_config import Yolov5TorchConfig
from .yolov5_models.models.experimental import attempt_load
from .yolov5_models.utils.datasets import letterbox
from .yolov5_models.utils.general import check_img_size, non_max_suppression_face,\
                                             scale_coords, xyxy2xywh

class YOLOV5Torch:
    def __init__(self, config: Yolov5TorchConfig) -> None:
        self.conf_thres = config.confidence_threshold
        self.iou_thres = config.iou_threshold
        self.device = config.device
        self.weight = config.weight
        self.model = attempt_load(self.weight, self.device)

    def image_processing(self, orgimg):
        img_size = 800
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found '
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img, orgimg


    def detect(self, orgimg):
        img, orgimg =  self.image_processing(orgimg)
        
        pred = self.model(img)[0]

        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        point = []
        kpss = []
        for i, det in enumerate(pred):
            gn = torch.tensor(orgimg.shape, device=self.device)[[1, 0, 1, 0]]
            gn_lks = torch.tensor(orgimg.shape, device=self.device)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
                for j in range(det.size()[0]):
                    xyxy2xywh(det[j, :4].view(1, 4))/gn
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    result = self.get_bbox(orgimg, xywh, conf, landmarks)
                    point.append(result[0])
                    kpss.append(result[1])
        # print(point)
        return np.array(point), np.array(kpss)

    def get_bbox(self, img, xyxy, conf, landmarks):
        h,w,c = img.shape
        x1 = int(xyxy[0]*w - xyxy[2]*w/2)
        y1 = int(xyxy[1]*h - xyxy[3]*h/2)
        x2 = int(xyxy[0]*w + xyxy[2]*w/2)
        y2 = int(xyxy[1]*h + xyxy[3]*h/2)
        det = [x1,y1,x2,y2,conf]
        kp = []
        for i in range(5):
            point_x = int(landmarks[2 * i]*w)
            point_y = int(landmarks[2 * i + 1]*h)
            kp.append([point_x, point_y])
            # result.append(point_x)
            # result.append(point_y)
        # result.append(float(str(conf)[:5]))
        return det, kp

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        #clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords
