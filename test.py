# from modules.utils.tensorrt_model import TensorrtInference
# from modules.face_detection.face_detection import FaceDetection
# from modules.face_embedding.face_embedding import FaceEmbedding
# from modules.utils.detection import img_vis
from modules.video_stream.video_reader import VideoReader
from queue import Queue

import cv2
import torch
import numpy as np
from skimage import transform as trans
import yaml
# face_det = FaceDetection("configs/config.yaml")
# img = cv2.imread("images/cr7.jpg")
# img,orgimg =  face_det.pre_process(img)
# pred = face_det.inference(img)
# pred = face_det.post_process(img,orgimg,pred)
# print(pred)
# # img_vis(img,orgimg,pred)


# src = np.array([
#                 [30.2946, 51.6963],
#                 [65.5318, 51.5014],
#                 [48.0252, 71.7366],
#                 [33.5493, 92.3655],
#                 [62.7299, 92.2041]], dtype=np.float32 )
# src[:,0] += 8.0
# print(src.shape[1])
# pred = pred[0]
# face = orgimg[pred[1]:pred[3], pred[0]:pred[2]]
# face = cv2.resize(face, (112, 112))
# # cv2.imwrite("image.jpg",face)
# dst = np.array(pred[4:14]).astype(np.float32)
# print(dst)
# tform = trans.SimilarityTransform()
# tform.estimate(dst, src)
# M = tform.params[0:2,:]
# warped = cv2.warpAffineconfig_fpcess(img)
# pred = face_det.inference(img)
# ts1 = torch.tensor(pred)

# img = cv2.imread("cr71.jpg")
# img=  face_det.pre_process(img)
# pred = face_det.inference(img)
# ts2 = torch.tensor(pred)

# cosi = torch.nn.CosineSimilarity(dim=0)
# output = cosi(ts1, ts2)
# print("Computed Cosine Similarity: ", output)

# with open("configs/config.yaml") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# video_reader_config = config["video_reader"]
# queue = Queue()
# video_reader = VideoReader(video_reader_config, queue)
# video_reader.run()



