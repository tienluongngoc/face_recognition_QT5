from inferences  import SCRFD, ArcFace, FAISS
from inferences.utils.face_detect import Face
from models.person import PersonDoc
from database import PersonDatabase
from configs import SCRFDConfig, ArcFaceConfig, FaissConfig, FaceRecogAPIConfig
import numpy as np
from typing import List

class FaceRecognitionApp:
    def __init__(self) -> None:
        global_config = FaceRecogAPIConfig(config_path="configs/face_recog_api.yaml")
        database = PersonDatabase(global_config.mongodb)
        self.face_detection = SCRFD(global_config.detection)
        self.face_encode = ArcFace(global_config.encode)
        self.recognizer = FAISS(global_config.recognition, database)
        self.recognizer.initialize()

    def encode(self, image: np.ndarray) -> np.ndarray:
        detection_results = self.face_detection.detect(image)
        if detection_results[0].shape[0] == 0 or detection_results[1].shape[0] == 0:
            return np.array([])
        detection_results = self.get_largest_bbox(detection_results)
        face = Face(
			bbox=detection_results[0][:4], 
			kps=detection_results[1][0], 
			det_score=detection_results[0][-1]
		)
        encode_results = self.face_encode.get(image, face)
        return encode_results
    
    def check_face_liveness(self, image):
        live = True
        return live

    def get_largest_bbox(self, detection_results: List[np.ndarray]) -> List[np.ndarray]:
        bboxes = detection_results[0]
        kpss = detection_results[1]
        largest_bbox = max(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        kps_corr = kpss[np.where((bboxes==largest_bbox).all(axis=1))]
        detection_largest = [largest_bbox, kps_corr]
        return detection_largest
    
    def recognize(self, image: np.ndarray):
        if image is None:
            raise
        embed_vector = self.encode(image)
        if embed_vector.size == 0:
            return "Nothing"

        state = ""
        face_live = self.check_face_liveness(image)
        if face_live is not None:
            if face_live:
                state = "real"
            else:
                state = "fake"

        person_info = self.recognizer.search(embed_vector)

        #TODO check

        person_doc = PersonDoc(id=person_info["person_id"], name=person_info["person_name"])
        person_dict = person_doc.dict()
        person_dict["state"] = state
        return person_dict