from configs import all_config
from .face_detection.scrfd import SCRFD
from .face_encode.arcface import ArcFace
from .face_recognition.faiss_wrap import FAISS, ChangeEvent
from .face_anti_spoofing.fasnet_emsemble import MiniFASNetEmsemble
from database import PersonDB_instance

det_infer = SCRFD(all_config.detection)
enc_infer = ArcFace(all_config.encode)
reg_infer = FAISS(all_config.recognition, local_db=PersonDB_instance)
reg_infer.initialize()
# anti_infer = MiniFASNetEmsemble([all_config.anti_spoofing_v1se, all_config.anti_spoofing_v2])
