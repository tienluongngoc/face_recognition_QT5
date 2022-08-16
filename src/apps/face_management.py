from distutils.command.config import config
from database import PersonDatabase
from models.person import PersonDoc
from validation import PersonVerify
import os
import shutil
import numpy as np
from fastapi import HTTPException, status
from schemas import ImageValidation,Validation
from models import EmbeddingVectorDoc, FaceDoc
from fastapi.responses import JSONResponse
import uuid
from validation import FaceValidation
from pathlib import Path
import cv2
from utils.utils import npfloat2float
from inferences import ChangeEvent, face_encode
from urllib.parse import unquote
from apps.face_recognition_factory import FaceRecognitionFactory
from configs.config_instance import FaceRecognitionConfigInstance
from .face_recognition import FaceRecognition

class FaceManagement(FaceRecognition):
    def __init__(self, face_config, db_instance: PersonDatabase) -> None:
        super(FaceManagement, self).__init__()
        self.face_config = face_config
        self.db_instance = db_instance
        config = FaceRecognitionConfigInstance.__call__().get_config()
        self.face_recognizer = FaceRecognitionFactory.__call__(config).get_engine()

        self.verify = PersonVerify(db_instance=db_instance)
        self.face_validation = FaceValidation(self.face_config,self.face_detection, self.face_encode, self.face_recognizer)

    def insert_face(self, person_id: str, face_id: str, image: np.ndarray):
        person_id, face_id = unquote(person_id), unquote(face_id)

        if not self.verify.check_person_by_id(person_id):
            return Validation.PERSON_ID_NOT_FOUND
        if self.verify.check_face_by_id(person_id, face_id):
            return Validation.FACE_ID_ALREADY_EXIST

        person_doc = self.db_instance.personColl.find_one(
            {"id": person_id}, {"_id": 0})

        image_validation = self.face_validation.validate_face(image, person_doc)
        if image_validation == ImageValidation.IMAGE_IS_VALID:
            vector = self.face_validation.encode(image)
            vector = npfloat2float(vector)
            embed_doc = EmbeddingVectorDoc(
                id=str(uuid.uuid4().hex),
                engine=self.face_validation.engine,
                value=vector
            )
            face_dir = os.path.join(self.face_config["path"], person_id)
            Path(face_dir).mkdir(parents=True, exist_ok=True)
            image_path = os.path.join(
                face_dir, f"{face_id}{self.face_config['end']}")
            cv2.imwrite(image_path, image)
            face_doc = FaceDoc(
                id=face_id, imgPath=image_path, vectors=[embed_doc])
            if "faces" not in person_doc.keys() or person_doc["faces"] is None:
                self.db_instance.personColl.update_one(
                    {"id": person_id}, {"$set": {"faces": [face_doc.dict()]}}
                )
            else:
                self.db_instance.personColl.update_one(
                    {"id": person_id}, {"$push": {"faces": face_doc.dict()}}
                )
            status_res = Validation.CREATED_FACE
            person_doc = self.db_instance.personColl.find_one(
                {"id": person_id}, {"_id": 0})
            self.face_recognizer.add_change_event(
                event=ChangeEvent.add_vector,
                params=[person_doc]
            )
        else:
            status_res = Validation.NOT_ACCEPTABLE

        return status_res, image_validation

    def select_all_face_of_person(self, person_id: str, skip: int, limit: int):
        if not self.verify.check_person_by_id(person_id):
            return Validation.PERSON_ID_NOT_FOUND
        person_doc = self.db_instance.personColl.find_one(
            {"id": person_id}, {"faces.vectors.value": 0})
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

    def delete_face_by_id(self, person_id: str, face_id: str):
        if not self.verify.check_face_by_id(person_id, face_id):
            return Validation.PERSON_ID_HAS_NOT_FACE_ID

        person_doc = list(self.db_instance.personColl.find({
            "$and": [
                {"id": person_id},
                {"faces.id": face_id}
            ]}, {"_id": 0, "faces.vectors.id": 1}
        ))
        if "vectors" in person_doc[0]["faces"][0].keys():
            if person_doc[0]["faces"][0]["vectors"] is not None:
                vector_ids = [v["id"]
                              for v in person_doc[0]["faces"][0]["vectors"]]
                for vector_id in vector_ids:
                    face_recognizer.add_change_event(
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
            return Validation.PERSON_ID_NOT_FOUND
        self.db_instance.personColl.update_one(
            {"id": person_id},
            {"$pull": {"faces": {}}}
        )
        image_dir = os.path.join(self.face_config["path"], person_id)
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)

        face_recognizer.add_change_event(
            event=ChangeEvent.remove_person,
            params=[person_id]
        )
