from .face_recognition_config import FaceRecogAPIConfig
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FaceRecognitionConfigInstance(metaclass=Singleton):
    def __init__(self) -> None:
        super(FaceRecognitionConfigInstance, self).__init__()
        self.config = FaceRecogAPIConfig("configs/face_recognition.yaml")
    
    def get_config(self):
        return self.config