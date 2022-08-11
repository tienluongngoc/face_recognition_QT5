from asyncio import tasks
from .face_recognition import FaceRecognition
from configs import FaceRecogAPIConfig
from database import PersonDatabase
from inferences import FAISS
from video_stream import VideoReader, video_reader
from .data_queue import DataQueue
import cv2


class FaceRecognitionApp:
    def __init__(self) -> None:
        global_config = FaceRecogAPIConfig(config_path="configs/face_recog_api.yaml")
        database = PersonDatabase(global_config.mongodb)
        self.recognizer = FAISS(global_config.recognition, database)
        self.recognizer.initialize()
        self.frame_queue = DataQueue.__call__().get_frame_queue()

        self.tasks = []
        face_recognizer = FaceRecognition(self.recognizer)
        self.tasks.append(face_recognizer)
        for i in range (1):
            video_reader = VideoReader("", self.frame_queue)
            self.tasks.append(video_reader)
            
            
    
    def run(self):
        for task in self.tasks:
            task.start()    