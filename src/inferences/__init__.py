from configs import all_config
from .face_detection.scrfd import SCRFD
from .face_encode.arcface import ArcFace
from .face_recognition.faiss_wrap import FAISS, ChangeEvent
from .face_anti_spoofing.fasnet_emsemble import MiniFASNetEmsemble
from database import PersonDB_instance
from .base import Singleton
from .tensorrt_base.trt_model import TRTModel
from .face_detection.yolov5 import YOLOV5
# import utils.yolov5_utils

face_detection = SCRFD(all_config.detection)
face_encode = ArcFace(all_config.encode)
# face_recognizer = FAISS(all_config.recognition, local_db=PersonDB_instance)
# face_recognizer.initialize()