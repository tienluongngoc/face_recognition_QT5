from distutils.command.config import config
from database import PersonDatabase
from models.person import PersonDoc
from services.validation import face_validation
from ..validation import PersonVerify
import os, shutil
import numpy as np
from fastapi import HTTPException, status
from schemas import ImageValidation
from models import EmbeddingVectorDoc, FaceDoc
from fastapi.responses import JSONResponse
import uuid
from ..validation import FaceValidation
from pathlib import Path
import cv2
from utils import npfloat2float
from inferences import reg_infer, ChangeEvent, enc_infer

class FaceCRUD:
	def __init__(self, face_config, db_instance: PersonDatabase) -> None:
		self.face_config = face_config
		self.db_instance = db_instance
		self.verify = PersonVerify(db_instance=db_instance)
		self.face_validation = FaceValidation(face_config)

	
	def insert_face(self, person_id: str, face_id: str, image: np.ndarray):
		if not self.verify.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		if self.verify.check_face_by_id(person_id, face_id):
			raise HTTPException(status.HTTP_409_CONFLICT)
		
		person_doc = self.db_instance.personColl.find_one({"id": person_id}, {"_id": 0})
		
		validate_result = self.face_validation.validate_face(image, person_doc)
		if validate_result == ImageValidation.IMAGE_IS_VALID:
			vector = self.face_validation.encode(image)
			vector = npfloat2float(vector)
			embed_doc = EmbeddingVectorDoc(
				id = str(uuid.uuid4().hex),
				engine=self.face_validation.engine,
				value=vector
			)
			face_dir = os.path.join(self.face_config["path"], person_id)
			Path(face_dir).mkdir(parents=True, exist_ok=True)
			image_path = os.path.join(face_dir, f"{face_id}{self.face_config['end']}")
			cv2.imwrite(image_path, image)
			face_doc = FaceDoc(id=face_id, imgPath=image_path, vectors=[embed_doc])
			if "faces" not in person_doc.keys() or person_doc["faces"] is None:
				self.db_instance.personColl.update_one(
					{"id": person_id}, {"$set": {"faces": [face_doc.dict()]}}
				)
			else:
				self.db_instance.personColl.update_one(
					{"id": person_id}, {"$push": {"faces": face_doc.dict()}}
				)
			status_res = status.HTTP_201_CREATED
			person_doc = self.db_instance.personColl.find_one({"id": person_id}, {"_id": 0})
			reg_infer.add_change_event(
				event=ChangeEvent.add_vector,
				params=[person_doc]
			)
		else:
			status_res = status.HTTP_406_NOT_ACCEPTABLE

		return JSONResponse(
			status_code=status_res,
			content={"INFO": validate_result}
		)

	def select_all_face_of_person(self, person_id: str, skip: int, limit: int):
		if not self.verify.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		person_doc = self.db_instance.personColl.find_one({"id": person_id}, {"faces.vectors.value": 0})
		if "faces" not in person_doc.keys() or person_doc["faces"] is None:
			return []

		faces = person_doc["faces"]
		if skip < 0:
				skip = 0
		elif skip > len(faces):
				skip = len(faces) - 1
		if limit < 0:
				limit = 0
		elif limit > len(faces) - skip:
				limit = len(faces) - skip
		faces = faces[skip: skip + limit]
		return faces

	def update_face_data(self):
		pass

	def update_image_path(self):
		pass

	def delete_face_by_id(self, person_id: str, face_id: str):
		if not self.verify.check_face_by_id(person_id, face_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		
		person_doc = list(self.db_instance.personColl.find({
			"$and": [
				{"id": person_id},
				{"faces.id": face_id}
			]}, {"_id": 0, "faces.vectors.id": 1}
		))
		if "vectors" in person_doc[0]["faces"][0].keys():
			if person_doc[0]["faces"][0]["vectors"] is not None:
				vector_ids = [v["id"] for v in person_doc[0]["faces"][0]["vectors"]]
				for vector_id in vector_ids:
					reg_infer.add_change_event(
						event=ChangeEvent.remove_vector,
						params=[vector_id]
					)

		self.db_instance.personColl.update_one(
			{"id": person_id, "faces.id": face_id},
			{"$pull": {"faces": {"id": face_id}}}
		)
		image_path = os.path.join(
			self.face_config["path"],
			person_id, face_id + self.face_config["end"]
		)
		if os.path.exists(image_path):
				os.remove(image_path)

	def delete_all_face(self, person_id: str):
		if not self.verify.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		self.db_instance.personColl.update_one(
			{"id": person_id},
			{"$pull": {"faces": {}}}
		)
		image_dir = os.path.join(self.face_config["path"], person_id)
		if os.path.exists(image_dir):
			shutil.rmtree(image_dir)
		
		reg_infer.add_change_event(
			event=ChangeEvent.remove_person,
			params=[person_id]
		)
	
	def recognize(self, image: np.ndarray):
		if image is None:
			raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE) 

		embed_vector = self.face_validation.encode(image)
		if embed_vector.size == 0:
			return JSONResponse(
				status_code=status.HTTP_406_NOT_ACCEPTABLE,
				content={"INFO": "This image has no face"}
			)

		state = ""
		face_live = self.face_validation.check_face_liveness(image)
		if face_live is not None:
			if face_live:
				state = "real"
			else:
				state = "fake"

		person_info = reg_infer.search(embed_vector)
		if len(person_info.keys()) == 0:
			return JSONResponse(
				status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
				content={"INFO": "Recognition Model is not ready."}
			)
		elif len(person_info.keys()) == 1:
			return JSONResponse(
				status_code=status.HTTP_404_NOT_FOUND,
				content={"INFO": "Unrecognize person in this image.", "state": state}
			)
		person_doc = PersonDoc(id=person_info["person_id"], name=person_info["person_name"])
		person_dict = person_doc.dict()
		person_dict["state"] = state
		return person_dict
		# return JSONResponse(
		# 		status_code=status.HTTP_200_OK,
		# 		content=person_dict
		# 	)		
