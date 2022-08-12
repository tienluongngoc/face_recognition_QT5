from pathlib import Path
import shutil
from fastapi import status, HTTPException
from src.database import PersonDatabase
from ..services.validation import PersonVerify
from schemas import SimplePerson
from models import PersonDoc
from inferences import face_recognizer, ChangeEvent
import os
from urllib.parse import unquote

class PersonManagement:
	def __init__(self, face_config, db_instance: PersonDatabase) -> None:
		self.face_config = face_config
		self.db_instance = db_instance
		self.verify = PersonVerify(db_instance=db_instance)
	
	def insert_person(self, id: str, name: str) -> PersonDoc:
		id,name = unquote(id), unquote(name)
		person = SimplePerson(id=id, name=name)
		if self.verify.check_person_by_id(person.id):
			raise 
		
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
			raise HTTPException(status.HTTP_404_NOT_FOUND)

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
			raise 
		self.db_instance.personColl.update_one(
			{"id": person_id},
			{"$set": {"name": name}}
		)

		face_recognizer.add_change_event(
			event=ChangeEvent.update_name,
			params=[person_id, name]
		)

	def update_person_id(self, person_id: str, new_id: str):
		person_id, new_id = unquote(person_id), unquote(new_id)
		if not self.verify.check_person_by_id(person_id):
			raise 
		self.db_instance.personColl.update_one(
			{"id": person_id},
			{"$set": {"id": new_id}}
		)
		current_images_path = os.path.join(
			self.face_config["path"], person_id)
		if os.path.exists(current_images_path):
			new_images_path = os.path.join(
				self.face_config["path"], new_id)
			os.rename(current_images_path, new_images_path)
		
		face_recognizer.add_change_event(
			event=ChangeEvent.update_id,
			params=[person_id, new_id]
		)

	def delete_person_by_id(self, id: str):
		if not self.verify.check_person_by_id(id):
			raise 
		image_dir = os.path.join(self.face_config["path"], id)
		if os.path.exists(image_dir):
			shutil.rmtree(image_dir)

		self.db_instance.personColl.delete_one({"id": id})

		face_recognizer.add_change_event(
			event=ChangeEvent.remove_person,
			params=[id]
		)

	def delete_all_people(self):
		self.db_instance.personColl.delete_many({})
		if os.path.exists(self.face_config["path"]):
			shutil.rmtree(self.face_config["path"])
			Path(self.face_config["path"]).mkdir(
				parents=True, exist_ok=True)
		
		face_recognizer.add_change_event(
			event=ChangeEvent.remove_all_db,
			params=[]
		)
