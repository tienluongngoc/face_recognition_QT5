from .person import PersonCRUD, FaceCRUD
from database import PersonDB_instance
from configs import faces_config

PersonCRUD_instance = PersonCRUD(face_config=faces_config, db_instance=PersonDB_instance)
FaceCRUD_instance = FaceCRUD(face_config=faces_config, db_instance=PersonDB_instance)