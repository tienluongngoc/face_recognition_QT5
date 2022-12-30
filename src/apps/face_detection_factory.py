from distutils.command.config import config
from src.inferences.face_detection.scrfd import SCRFD
from src.configs.yolov5_config import Yolov5Config
from src.configs.scrfd_config import SCRFDConfig
from src.configs.face_recognition_config import FaceRecogAPIConfig

class FaceDetectionFactory:
    def __init__(self, config:FaceRecogAPIConfig) -> None:
        self.engine_name = config.detection_engine
        self.detection_config = config.detection

    def get_engine(self):
        if self.engine_name == "scrfd":
            engine = SCRFD(self.detection_config)
        elif self.engine_name == "yolov5":
            from src.inferences.face_detection.yolov5 import YOLOV5
            engine = YOLOV5(self.detection_config)
        # elif self.engine_name == "yolov5_torch":
        #     engine = YOLOV5Torch(self.detection_config)
        return engine