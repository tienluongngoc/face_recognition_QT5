from ..tensorrt_base.trt_model import TRTModel
from configs import ArcFaceTRTConfig
from ..utils import face_align, face_detect
import torchvision
import copy
import cv2
import torch
import time
import numpy as np

class ArceFaceTRT(TRTModel):
    def __init__(self, config: ArcFaceTRTConfig):
        super().__init__(config)
        self.output_shape = config.output_shape
        self.input_shape = tuple([config.input_shape[2], config.input_shape[3]])
        self.stride_max = config.stride_max
        self.confidence_threshold = config.confidence_threshold
        self.iou_threshold = config.iou_threshold

    def get(self, img: np.ndarray, face: face_detect.Face):
        aimg = face_align.norm_crop(img, landmark=face.kps)
        face.embedding = self.get_feat(aimg).flatten()
        return face.normed_embedding

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_shape
        blob = cv2.dnn.blobFromImages(
			imgs, 1.0 / self.config.input_std, input_size,
			(self.config.input_mean, self.config.input_mean, self.config.input_mean), swapRB=True
		)
        blob = torch.from_numpy(blob).to("cuda")
        embedding = self(blob)
        embedding = embedding.cpu().detach().numpy()
        return embedding
