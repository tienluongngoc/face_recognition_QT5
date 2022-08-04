
from modules.face_detection.face_detection import FaceDetection
from modules.face_embedding.face_embedding import FaceEmbedding
from modules.database.database import Database

from threading import Thread
import cv2

class FaceRecognition(Thread):
    def __init__(self, config_fp, frame_queue) -> None:
        Thread.__init__(self)
        super(FaceRecognition, self).__init__()
        self.face_detection = FaceDetection(config_fp)
        self.face_embedding = FaceEmbedding(config_fp)
        # self.database = Database(config_fp)
        self.frame_queue = frame_queue
        self.is_stop = False
        

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
                    print(res.shape)
                    cv2.imwrite("img.jpg", face)
                

                # print(image)
                