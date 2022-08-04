from asyncio import tasks
from modules.face_recognition.face_recognition import FaceRecognition
from modules.video_stream.video_reader import VideoReader
import yaml
from queue import Queue
# with open("configs/config.yaml") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# video_reader_config = config["video_reader"]
# queue = Queue()
# video_reader = VideoReader(video_reader_config, queue)
# video_reader.run()

class FaceRecognitionApp:
    def __init__(self, config_fp) -> None:
        with open("configs/config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.frame_queue = Queue()
        self.video_reader = VideoReader(config["video_reader"], self.frame_queue)
        self.face_recognizer = FaceRecognition(config_fp, self.frame_queue)
        
        self.tasks = []
        self.tasks.append(self.video_reader)
        self.tasks.append(self.face_recognizer)
    
    def start(self):
        for task in self.tasks:
            task.start()


app = FaceRecognitionApp("configs/config.yaml")
app.start()