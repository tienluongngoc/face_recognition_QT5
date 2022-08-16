from .face_recognition import FaceRecognition
from .face_recognition_app import FaceRecognitionApp
from .data_queue import DataQueue
from .person_management import PersonManagement
from .face_management import FaceManagement
from .face_recognition_factory import FaceRecognitionFactory
from configs import FaceRecognitionConfigInstance
from database import DatabaseInstance

database_instance = DatabaseInstance.__call__().get_database()
config = FaceRecognitionConfigInstance.__call__().get_config()
face_config = config.faces
person_management_instance = PersonManagement(face_config=face_config,db_instance=database_instance)
face_management_instance = FaceManagement(face_config=face_config, db_instance=database_instance)
