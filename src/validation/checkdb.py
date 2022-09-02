from src.database.PersonDB import PersonDatabase

class PersonVerify:
	def __init__(self, db_instance: PersonDatabase):
		self.db_instance = db_instance
	
	def check_person_by_id(self, person_id: str) -> bool:
		person_doc = self.db_instance.personColl.find_one({"id": person_id}, {"_id": 0})
		if person_doc is None:
			return False
		return True
	
	def check_face_by_id(self, person_id: str, face_id: str) -> bool:
		person_doc = self.db_instance.personColl.find_one({"id": person_id}, {"_id": 0, "faces.vectors": 0})
		if person_doc is None:
			return False
		if "faces" not in person_doc.keys():
			return False
		elif person_doc["faces"] is None:
			return False
		elif len(person_doc["faces"]) == 0:
			return False
		current_ids = [x["id"] for x in person_doc["faces"]]
		if face_id not in current_ids:
			return False
		return True
		
	def check_embed_by_id(self, person_id: str, face_id: str, embed_id: str) -> bool:
		person_doc = self.Persondb.personColl.find_one(
			{"id": person_id}, {"_id": 0}
		)
		if person_doc is None:
			return False
		if "faces" not in person_doc.keys():
			return False
		elif person_doc["faces"] is None:
			return False
		elif len(person_doc["faces"]) == 0:
			return False
		curent_face_ids = [x["id"] for x in person_doc["faces"]]
		if face_id not in curent_face_ids:
			return False
		
		embed_vectors = []
		for face in person_doc["faces"]:
			if "vectors" not in person_doc.keys() or len(person_doc["vectors"]) == 0:
				continue
			embed_vectors += [vector for vector in face["vectors"]]

		if len(embed_vectors) == 0:
			return False
		current_embed_ids = [vector["id"] for vector in embed_vectors]
		if embed_id not in current_embed_ids:
			return False
		return True
