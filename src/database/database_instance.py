# from ..utils.utils import Singleton
from .PersonDB import PersonDatabase
from src.configs.config_instance import FaceRecognitionConfigInstance

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
        
class DatabaseInstance(metaclass=Singleton):
    def __init__(self) -> None:
        super(DatabaseInstance, self).__init__()
        config = FaceRecognitionConfigInstance.__call__().get_config()
        database_config = config.mongodb
        self.database_instance = PersonDatabase(database_config)
    
    def get_database(self):
        return self.database_instance