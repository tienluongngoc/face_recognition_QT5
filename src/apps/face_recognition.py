from os import pread
from unittest import skip
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
import time

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
        self.tracked_face = {}
    
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
        print(self.tracked_face)
        # print(bboxes)
        # print(kpss)
        ids = {}
        # print("------------------------")
        if len(bboxes) != 0:
            tracks = self.tracker.update(bboxes)
            # print(tracks)
            for i,bbox in enumerate(bboxes):
                for j,track in enumerate(tracks):
                    bb1 = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    bb2 = [track[0], track[1],track[2],track[3]]
                    iou_result = self.iou(bb1, bb2)
                    if iou_result > 0.9:
                        ids[f"{i}"] = track[4]
        new_bboxes = []
        for key in ids.keys():
            if ids[key] not in self.tracked_face:
                self.tracked_face[int(ids[key])] = {"id": [], "skiped": 0, "time": time.time()}
        

        # print(ids)
        # print(bboxes)
        for i,bbox in enumerate(bboxes):
            if str(i) in ids.keys():
                new_bbox = [bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],int(ids[f"{i}"])]
            # else:
            #     new_bbox = [bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],-1]
                new_bboxes.append(new_bbox)
        # test = []
        # for i,bbox in enumerate(new_bboxes):
        #     test.append([bbox, kpss[i]])
        # print(len(test))
        detection_results = (np.array(new_bboxes), kpss)
        print(detection_results)

        if detection_results[0].shape[0] == 0 or detection_results[1].shape[0] == 0:
            encode_results = {}
            return (np.array([]), {})
        inference_on_largest_bbox = False
        encode_results = {}
        if inference_on_largest_bbox:
            largest_bbox = self.get_largest_bbox(detection_results)
            face = Face(bbox=largest_bbox[0][:4], kps=largest_bbox[1][0],det_score=largest_bbox[0][-1])
            encode_result = self.face_encode.get(image, face)
            encode_results.append(encode_result)
            return largest_bbox,encode_results
        else:
            for i,detection_result in enumerate(detection_results[0]):
                # if (int(detection_result[5]) not in self.tracked_face.keys):
                face_id = int(detection_result[5])
                if face_id in self.tracked_face.keys():
                    skiped = self.tracked_face[face_id]["skiped"]
                    if skiped > 5 or skiped == 0:
                        # print("oke")
                        # print(detection_result[:4])
                        # print(detection_results[1][i])
                        face = Face(bbox=detection_result[:4], kps=detection_results[1][i],det_score=detection_result[-2])
                        encode_result = self.face_encode.get(image, face)
                        # print(encode_result[:10])
                        # encode_results.append(encode_result)
                        encode_results[face_id] = encode_result
                        self.tracked_face[face_id]["skiped"] = 1
                    else:
                        self.tracked_face[face_id]["skiped"] = self.tracked_face[face_id]["skiped"] + 1
                # else:
                #     pass

                # print(len(detection_results), len(encode_results))

            return detection_results, encode_results
        
    
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

    def clean_cache(self):
        del_id_list = []
        for key in self.tracked_face.keys():
            # remove last 20 frame, keep cache 20
            if len(self.tracked_face[key]["id"]) > 15:
                self.tracked_face[key]["id"].pop(0)

            # remove face id if 3s was not update, keep 3s
            if (time.time() - self.tracked_face[key]["time"]) > 3:
                del_id_list.append(key)
        for id in del_id_list:
            del self.tracked_face[id]
        # print(del_id_list)

    def most_frequent(self, List):
        return max(set(List), key = List.count)

    
    def run(self):
        while True:
            self.clean_cache()
            image =  self.frame_queue.get()["image"]
            result = {}
            if self.recognize:
                print("--------------------------------------------------------------------------------------------------------------------------------------")
                if image is None:
                    raise
                detection_results,embed_vectors = self.encode(image)
                result["detection_results"] = detection_results
                person_dicts = []
                for key in embed_vectors.keys():
                    embed_vector = embed_vectors[key]
                    if embed_vector.size != 0:
                        state = ""
                        face_live = self.check_face_liveness(image)
                        if face_live is not None:
                            if face_live:
                                state = "real"
                            else:
                                state = "fake"
                        # print("search")
                        person_info = self.recognizer.search(embed_vector)
                        # print(person_info)

                        # #TODO check
                        if person_info["person_id"] != "unrecognize":
                            person_doc = PersonDoc(id=person_info["person_id"], name=person_info["person_name"])
                            person_dict = person_doc.dict()
                            person_dict["state"] = state
                            #TODO check liveness
                        else:
                            person_dict = {"id": "unknown"}
                        # print(person_dict)
                        # person_dicts.append(person_dict)
                        # face_id = detection_results[0][i][5]
                        print("------------>", key)
                        self.tracked_face[key]["id"].append(person_dict["id"])
                        self.tracked_face[key]["time"] = time.time()
                # result["person_dict"] = person_dicts
                # print(detection_results)
                if len(detection_results) != 0:
                    bboxes = detection_results[0]
                    # print(bboxes)
                    for bbox in bboxes:
                        face_id = int(bbox[5])
                        if face_id in self.tracked_face.keys():
                            id_list = self.tracked_face[face_id]["id"]
                            # print(id_list)
                            if len(id_list):
                                id = self.most_frequent(id_list)
                            else:
                                id = "unknown"
                            person_dicts.append({"id": id, "number_frame": len(id_list)})

                # print(bboxes)
                
                print(self.tracked_face)
                result["person_dict"] = person_dicts
            result["image"] = image
            
            self.result_queue.put(result)

    def search(self, embed_vector, result):
        pass


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