
from src.apps.face_recognition_factory import FaceRecognitionFactory
from src.configs.config_instance import FaceRecognitionConfigInstance
from src.inferences.face_recognition.faiss_wrap import ChangeEvent
from src.database.PersonDB import PersonDatabase
from src.validation.checkdb import PersonVerify
from src.schemas.validation import Validation
from src.schemas.person import SimplePerson
from src.models.person import PersonDoc
from urllib.parse import unquote
from pathlib import Path
import shutil
import os

class PersonManagement:
	def __init__(self, face_config, db_instance: PersonDatabase) -> None:
		self.face_config = face_config.faces
		self.db_instance = db_instance
		self.verify = PersonVerify(db_instance=db_instance)
		config = FaceRecognitionConfigInstance.__call__().get_config()
		self.face_recognizer = FaceRecognitionFactory.__call__(config).get_engine()
	
	def insert_person(self, id: str, name: str) -> PersonDoc:
		id,name = unquote(id), unquote(name)
		person = SimplePerson(id=id, name=name)
		if self.verify.check_person_by_id(person.id):
			return Validation.PERSON_ID_ALREADY_EXIST 
		
		person_doc = PersonDoc(id=person.id, name=person.name)
		self.db_instance.personColl.insert_one(person_doc.dict())
		return person_doc
	
	def select_all_people(self, skip: int, limit: int, have_vector: bool = False) -> list:
		listPeople = []
		if have_vector:
			docs = self.db_instance.personColl.find(
				{}, {"_id": 0}).skip(skip).limit(limit)
		else:
			docs = self.db_instance.personColl.find(
				{}, {"_id": 0, "faces.vectors": 0}).skip(skip).limit(limit)

		for doc in docs:
			listPeople.append(doc)

		return listPeople

	def select_person_by_id(self, person_id: str, have_vector: bool = False):
		if not self.verify.check_person_by_id(person_id):
			return Validation.PERSON_ID_NOT_FOUND

		if have_vector:
			doc = self.db_instance.personColl.find_one(
				{"id": person_id}, {"_id": 0}
			)
		else:
			doc = self.db_instance.personColl.find_one(
				{"id": person_id}, {"_id": 0, "faces.vectors": 0}
			)
		return doc

	def update_person_name(self, person_id: str, name: str):
		person_id, name = unquote(person_id), unquote(name)
		if not self.verify.check_person_by_id(person_id):
			return Validation.PERSON_ID_NOT_FOUND 
		self.db_instance.personColl.update_one(
			{"id": person_id},
			{"$set": {"name": name}}
		)
		self.face_recognizer.add_change_event(
			event=ChangeEvent.update_name,
			params=[person_id, name]
		)
		return Validation.UPDATE_PERSON_NAME

	def update_person_id(self, person_id: str, new_id: str):
		person_id, new_id = unquote(person_id), unquote(new_id)
		if not self.verify.check_person_by_id(person_id):
			return Validation.PERSON_ID_NOT_FOUND 
		person = self.select_person_by_id(person_id)
		if person["faces"] is not None:
			for face in person["faces"]:
				face["imgPath"] = face["imgPath"].replace(person_id, new_id)
		self.db_instance.personColl.update_one(
			{"id": person_id},
			{"$set": {"id": new_id}})

		self.db_instance.personColl.update_one(
			{"id": new_id},
			{"$set": {"faces": person["faces"]}}
		)

		current_images_path = os.path.join(self.face_config["path"], person_id)
		if os.path.exists(current_images_path):
			new_images_path = os.path.join(
				self.face_config["path"], new_id)
			os.rename(current_images_path, new_images_path)
		
		self.face_recognizer.add_change_event(
			event=ChangeEvent.update_id,
			params=[person_id, new_id]
		)
		return Validation.UPDATE_PERSON_ID

	def delete_person_by_id(self, id: str):
		if not self.verify.check_person_by_id(id):
			return  Validation.PERSON_ID_NOT_FOUND 
		image_dir = os.path.join(self.face_config["path"], id)
		if os.path.exists(image_dir):
			shutil.rmtree(image_dir)

		self.db_instance.personColl.delete_one({"id": id})

		self.face_recognizer.add_change_event(
			event=ChangeEvent.remove_person,
			params=[id]
		)
		return Validation.DETETE_PERSON_SUCCESSFULY

	def delete_all_people(self):
		self.db_instance.personColl.delete_many({})
		if os.path.exists(self.face_config["path"]):
			shutil.rmtree(self.face_config["path"])
			Path(self.face_config["path"]).mkdir(
				parents=True, exist_ok=True)
		
		self.face_recognizer.add_change_event(
			event=ChangeEvent.remove_all_db,
			params=[]
		)
