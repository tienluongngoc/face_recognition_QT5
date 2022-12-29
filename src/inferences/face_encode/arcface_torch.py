# from msilib.schema import Class
import numpy as np
import cv2
from ..utils import face_align, face_detect
from datetime import datetime
from ..base import Singleton
from src.configs.arcface_torch_config import ArcFaceTorchConfig
from .arcface_ultils.models import get_model
import torch


class ArcFaceTorch(metaclass=Singleton):
    def __init__(self, config: ArcFaceTorchConfig) -> None:
        self.config = config
        self.device = config.device
        self.weight = config.weight
        self.backbone = config.backbone
        self.model = get_model(self.backbone, fp16=False).to(self.device)
        self.model.load_state_dict(torch.load(self.weight))
        self.model.eval()
        self.model.to(self.device)

    def get(self, img: np.ndarray, face: face_detect.Face):
        aimg = face_align.norm_crop(img, landmark=face.kps)
        face.embedding = self.get_feat(aimg).flatten()
        return face.normed_embedding

    def get_feat(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        img = torch.tensor(img, device=self.device).float()

        feat = self.model(img)
        feat = feat.cpu().detach().numpy()
        return feat
        
    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def is_same(self, feat1, feat2):
        sim = self.compute_sim(feat1, feat2)
        if sim > self.config.threshold:
            return True
        else:
            return False