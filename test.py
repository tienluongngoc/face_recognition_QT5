from modules.utils.tensorrt_model import TensorrtInference
from modules.face_detection.face_detection import FaceDetection
from modules.face_embedding.face_embedding import FaceEmbedding
from modules.utils.detection import img_vis

import cv2
import torch

face_det = FaceDetection("configs/config.yaml")
img = cv2.imread("sample.jpg")
img,orgimg =  face_det.pre_process(img)
pred = face_det.inference(img)
pred = face_det.post_process(img,orgimg,pred)
print(pred)
# img_vis(img,orgimg,pred)


# face_det = FaceEmbedding("configs/config.yaml")

# img = cv2.imread("cr71.jpg")
# img=  face_det.pre_process(img)
# pred = face_det.inference(img)
# ts1 = torch.tensor(pred)

# img = cv2.imread("cr71.jpg")
# img=  face_det.pre_process(img)
# pred = face_det.inference(img)
# ts2 = torch.tensor(pred)

# cosi = torch.nn.CosineSimilarity(dim=0)
# output = cosi(ts1, ts2)
# print("Computed Cosine Similarity: ", output)
