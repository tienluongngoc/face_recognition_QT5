from os import pread
from .face_recognition_factory import FaceRecognitionFactory
from inferences import face_detection, face_encode
# from inferences import YOLOV5, SCRFD, ArcFace
from .face_detection_factory import FaceDetectionFactory
from .face_encode_factory import FaceEncodeFactory
from inferences.utils.face_detect import Face
from models.person import PersonDoc
from .data_queue import DataQueue, ResultQueue
import numpy as np
from typing import List
from threading import Thread
import cv2
# from configs import all_config
from configs import FaceRecognitionConfigInstance


class FaceRecognition(Thread):
    def __init__(self) -> None:
        Thread.__init__(self)
        super(FaceRecognition, self).__init__()
        self.frame_queue = DataQueue.__call__().get_frame_queue()
        self.result_queue = ResultQueue.__call__().get_result_queue()
        face_recognition_config = FaceRecognitionConfigInstance.__call__().get_config()
        self.face_detection = FaceDetectionFactory(face_recognition_config).get_engine()
        self.face_encode = FaceEncodeFactory(face_recognition_config).get_engine()
        self.recognizer = FaceRecognitionFactory.__call__(face_recognition_config).get_engine()
        self.recognizer.initialize()
        self.recognize = False
    
    def enable(self):
        self.recognize = True

    def disable(self):
        self.recognize = False
    
    def is_enable(self):
        return self.recognize

    def encode(self, image: np.ndarray) -> np.ndarray:
        detection_results = self.face_detection.detect(image)
        if detection_results[0].shape[0] == 0 or detection_results[1].shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        argest_bbox = self.get_largest_bbox(detection_results)
        face = Face(
			bbox=argest_bbox[0][:4], 
			kps=argest_bbox[1][0], 
			det_score=argest_bbox[0][-1]
		)
        encode_results = self.face_encode.get(image, face)
        return detection_results,argest_bbox,encode_results
    
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
    
    def run(self):
        while True:
            image =  self.frame_queue.get()["image"]
            result = {}
            if self.recognize:
                if image is None:
                    raise
                detection_results,largest_bbox,embed_vector = self.encode(image)
                result["detection_results"] = detection_results
                result["largest_bbox"] = largest_bbox
                if embed_vector.size != 0:
                    state = ""
                    face_live = self.check_face_liveness(image)
                    if face_live is not None:
                        if face_live:
                            state = "real"
                        else:
                            state = "fake"
                    person_info = self.recognizer.search(embed_vector)

                    # #TODO check
                    if person_info["person_id"] != "unrecognize":
                        person_doc = PersonDoc(id=person_info["person_id"], name=person_info["person_name"])
                        person_dict = person_doc.dict()
                        person_dict["state"] = state
                    else:
                        person_dict = {"id": "unknown"}
                    result["person_dict"] = person_dict
            
            result["image"] = image
            self.result_queue.put(result)
