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
from tracker import Sort


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
        self.tracker = Sort()
        self.recognize = False
    
    def enable(self):
        self.recognize = True

    def disable(self):
        self.recognize = False
    
    def is_enable(self):
        return self.recognize

    def encode(self, image: np.ndarray) -> np.ndarray:
        detection_results = self.face_detection.detect(image)
        bboxes = detection_results[0]
        kpss = detection_results[1]
        ids = {}
        if len(bboxes) != 0:
            tracks = self.tracker.update(bboxes)
            for i,bbox in enumerate(bboxes):
                for j,track in enumerate(tracks):
                    bb1 = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    bb2 = [track[0], track[1],track[2],track[3]]
                    iou_result = self.iou(bb1, bb2)
                    if iou_result > 0.9:
                        ids[f"{i}"] = track[4]
        new_bboxes = []
        for i,bbox in enumerate(bboxes):
            if str(i) in ids.keys():
                new_bbox = [bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],int(ids[f"{i}"])]
            else:
                new_bbox = [bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],-1]
            new_bboxes.append(new_bbox)
        detection_results = (np.array(new_bboxes), kpss)


        if detection_results[0].shape[0] == 0 or detection_results[1].shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        largest_bbox = self.get_largest_bbox(detection_results)
        face = Face(
			bbox=largest_bbox[0][:4], 
			kps=largest_bbox[1][0], 
			det_score=largest_bbox[0][-1]
		)
        encode_results = self.face_encode.get(image, face)
        return detection_results,largest_bbox,encode_results
    
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

    def iou(self, bb_test,bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return(o)