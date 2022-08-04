
from modules.face_detection.face_detection import FaceDetection
from modules.face_embedding.face_embedding import FaceEmbedding
from modules.database.database import Database
import torch

from threading import Thread
import cv2
import time
import os

class FaceRecognition(Thread):
    def __init__(self, config_fp, frame_queue) -> None:
        Thread.__init__(self)
        super(FaceRecognition, self).__init__()
        self.face_detection = FaceDetection(config_fp)
        self.face_embedding = FaceEmbedding(config_fp)
        self.database = Database.__call__(config_fp).get_connection()
        self.frame_queue = frame_queue
        self.is_stop = False

        self.face_embedding_collection = self.database["face_embedding"]
        

        # img_face = cv2.imread("images/face.jpg")
        # pre_face = self.face_embedding.pre_process(img_face)
        # face_emb =  self.face_embedding.inference(pre_face)
        # self.ts2 = torch.tensor(face_emb)
        self.cosi = torch.nn.CosineSimilarity(dim=0)
        
            
    def recognizer(self, infe_embedding):
        face_embedding_documents = self.face_embedding_collection.find()
        for person in face_embedding_documents:
            for db_embedding in person["embedding"]:
                db_embedding = torch.tensor(db_embedding)
                output = self.cosi(db_embedding, infe_embedding)

    def run(self):
        while True:
            if self.frame_queue.qsize():
                frame_data = self.frame_queue.get()
                image = frame_data["image"]
                image,org_image =  self.face_detection.pre_process(image)
                pred = self.face_detection.inference(image)
                pred_objects = self.face_detection.post_process(image,org_image,pred)
                for pred_object in pred_objects:
                    face = org_image[pred_object["bbox"][1]:pred_object["bbox"][3],pred_object["bbox"][0]:pred_object["bbox"][2]]
                    pre_face = self.face_embedding.pre_process(face)
                    res =  self.face_embedding.inference(pre_face)
                    infe_embedding = torch.tensor(res.tolist())
                    # print(infe_embedding)
                    self.recognizer(infe_embedding)
                    
                    # print(res)
                    # cv2.imwrite("img.jpg", face)
                # print(image)

    def insert_people(self):
        face_image = "face_images"
        for dir in os.listdir(face_image):
            dir_path = os.path.join(face_image, dir)
            # TODO check id exist in DB
            face_embedding_list = []
            for fn in os.listdir(dir_path):
                image = cv2.imread(os.path.join(dir_path, fn))
                image,org_image =  self.face_detection.pre_process(image)
                pred = self.face_detection.inference(image)
                pred_objects = self.face_detection.post_process(image,org_image,pred)
                if len(pred_objects) > 1:
                    continue
                for pred_object in pred_objects:
                    face = org_image[pred_object["bbox"][1]:pred_object["bbox"][3],pred_object["bbox"][0]:pred_object["bbox"][2]]
                    pre_face = self.face_embedding.pre_process(face)
                    face_embedding =  self.face_embedding.inference(pre_face)
                    # print(face_embedding.tolist())
                    face_embedding_list.append(face_embedding.tolist())
            if len (face_embedding_list):
                self.face_embedding_collection.insert_one({"person_id": f"{dir}","embedding": face_embedding_list})
            print(len(face_embedding_list))               