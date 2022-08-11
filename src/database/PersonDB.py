from .database import BaseDatabase
from src.models import PersonDoc
import numpy as np

class PersonDatabase(BaseDatabase):
	def __init__(self, config):
		super(PersonDatabase, self).__init__(config)
		self.db_name = config["person_representationdb"]["name"]
		self.coll_name = config["person_representationdb"]["person_coll"]["name"]
		self.db = self.client[self.db_name]
		self.personColl = self.db[self.coll_name]

		if config["save_db_local"]:
			self.save_db_local = True
		else:
			self.save_db_local = False
	
	def initialize_local_db(self):
		person_db = self.personColl.find(
			{}, {"_id": 0, "faces.id": 0, "faces.imgPath": 0, "faces.vectors.engine": 0}
		)
		self.vectors = {}
		self.people = {}
		for person in person_db:
			person_id = person["id"]
			person_name = person["name"]
			if ("faces" not in person.keys()) or (person["faces"] is None):
				continue
			for face in person["faces"]:
				if ("vectors" not in face.keys()) or (face["vectors"] is None):
					continue
				for vector in face["vectors"]:
					vector_id = vector["id"]
					value = np.array(vector["value"], dtype=np.float32)
					self.vectors[vector_id] = value / np.linalg.norm(value)
					self.people[vector_id] = {
						"person_id": person_id,
						"person_name": person_name
					}

	def get_local_db(self):
		return self.people, self.vectors

	def remove_person_from_local_db(self, person_id: str):
		keys_remove = [
			vector_id for vector_id in self.people.keys() 
			if self.people[vector_id]["person_id"] == person_id
		]
		for key in keys_remove:
			del self.people[key]
			del self.vectors[key] 

	def remove_vector_from_local_db(self, vector_id: str):
		if vector_id in self.people.keys():
			del self.people[vector_id]
		if vector_id in self.vectors.keys():
			del self.vectors[vector_id]

	def update_name_to_local_db(self, person_id: str, name: str):
		for key in self.people.keys():
			if self.people[key]["person_id"] == person_id:
				self.people[key]["person_name"] = name

	def update_id_to_local_db(self, current_person_id: str, new_person_id: str):
		for key in self.people.keys():
			if self.people[key]["person_id"] == current_person_id:
				self.people[key]["person_id"] = new_person_id

	def add_vector_to_local_db(self, person: PersonDoc):
		person_id = person["id"]
		person_name = person["name"]
		for face in person["faces"]:
			if ("vectors" not in face.keys()) or (len(face["vectors"]) == 0):
				continue
			for vector in face["vectors"]:
				vector_id = vector["id"]
				value = np.array(vector["value"], dtype=np.float32)
				self.vectors[vector_id] = value / np.linalg.norm(value)
				self.people[vector_id] = {
					"person_id": person_id,
					"person_name": person_name
				}

	def update_vector(self, vector_id, new_value):
		value = np.array(new_value, dtype=np.float32)
		self.vectors[vector_id] = value / np.linalg.norm(value)
	
	def remove_all_db(self):
		self.people = {}
		self.vectors = {}
