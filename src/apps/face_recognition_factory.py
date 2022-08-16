from inferences import FAISS
from configs.face_recognition_config import FaceRecogAPIConfig
from database.database_instance import DatabaseInstance
from utils.utils import Singleton

class FaceRecognitionFactory(metaclass=Singleton):
    def __init__(self, config:FaceRecogAPIConfig) -> None:
        super(FaceRecognitionFactory, self).__init__()
        
        self.engine_name = config.recognition_engine
        self.encode_config = config.recognition
        self.database_instance = DatabaseInstance.__call__().get_database()

    def get_engine(self):
        if self.engine_name == "faiss":
            engine = FAISS(self.engine_name, self.database_instance)
        return engine