from urllib.parse import unquote
from fastapi import APIRouter, status, Response
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from models.person import PersonDoc
from schemas.validation import ImageValidation
from x import FaceCRUD_instance
import cv2
import numpy as np
from models import FaceDoc
from typing import List

router = APIRouter(prefix="/people", tags=['faces'])

@router.get(
	"/{person_id}/faces",
	response_model= List[FaceDoc],
	status_code=status.HTTP_200_OK
)
async def get_all_faces(person_id: str, skip: int = 0, limit: int = 10):
	face_docs = FaceCRUD_instance.select_all_face_of_person(person_id, skip, limit)
	return face_docs

@router.post(
	"/{person_id}/faces",
	response_model=ImageValidation,
	status_code=status.HTTP_201_CREATED
)
async def insert_one_face(person_id: str, face_id: str, image: UploadFile = File(...)):
	content = await image.read()
	image_buffer = np.frombuffer(content, np.uint8)
	np_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
	person_id, face_id = unquote(person_id), unquote(face_id)
	res = FaceCRUD_instance.insert_face(person_id, face_id, np_image)
	return res


@router.delete(
	"/{person_id}/faces/{face_id}",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
async def delete_one_face(person_id: str, face_id: str):
	FaceCRUD_instance.delete_face_by_id(person_id, face_id)

@router.delete(
	"/{person_id}/faces",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
def delete_all_faces(person_id: str):
	FaceCRUD_instance.delete_all_face(person_id)

@router.post(
	"/recognition",
	response_model= PersonDoc,
	status_code=status.HTTP_200_OK
)
async def recognize_person(image: UploadFile = File(...)):
	content = await image.read()
	image_buffer = np.frombuffer(content, np.uint8)
	np_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
	person_doc = FaceCRUD_instance.recognize(np_image)
	return person_doc