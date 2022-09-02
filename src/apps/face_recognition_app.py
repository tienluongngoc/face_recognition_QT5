from asyncio import tasks
from .face_recognition import FaceRecognition
from src.configs.face_recognition_config import FaceRecogAPIConfig
from src.database.PersonDB import PersonDatabase
from src.inferences.face_recognition.faiss_wrap import FAISS
from src.video_stream.video_reader import VideoReader
from .data_queue import DataQueue
import cv2


class FaceRecognitionApp:
    def __init__(self) -> None:
        self.frame_queue = DataQueue.__call__().get_frame_queue()
        
        self.tasks = {}
        face_recognizer = FaceRecognition()
        self.tasks["face_recognizer"] = face_recognizer
        for i in range (1):
            video_reader = VideoReader("", self.frame_queue)
            self.tasks[f"video_reader_{i}"] = video_reader
            
    def run(self):
        for key in self.tasks.keys():
            self.tasks[key].start()
        
    def get_task(self):
        return self.tasks