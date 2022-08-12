from .face_recognition import FaceRecognition
from .face_recognition_app import FaceRecognitionApp
from .data_queue import DataQueue
from .person_management import PersonManagement
from .face_management import FaceManagement
from configs import faces_config
from database import PersonDB_instance

person_management_instance = PersonManagement(face_config=faces_config,db_instance=PersonDB_instance)
face_management_instance = FaceManagement(face_config=faces_config, db_instance=PersonDB_instance)