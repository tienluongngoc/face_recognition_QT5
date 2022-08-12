import enum
from http.client import CONFLICT, NOT_ACCEPTABLE

class ImageValidation(str, enum.Enum):
	"""
	Define face validation results:	\n
		IMAGE_IS_VALID: image is valid and was added to collection \n
		IMAGE_HAS_ALREADY_IN_DB: image has already in collection	\n
		IMAGE_HAS_NO_FACE: there isn't any faces in image \n
		FACE_NOT_OF_PERSON: Face in the image is different from those already in the collection \n
		INPUT_IS_NOT_IMAGE_FILE: Format of input from client is not image 
	"""
	IMAGE_IS_VALID = 'IMAGE_IS_VALID'
	IMAGE_HAS_ALREADY_IN_DB = "IMAGE_HAS_ALREADY_IN_DB"
	IMAGE_HAS_NO_FACE = "IMAGE_HAS_NO_FACE"
	FACE_NOT_OF_PERSON = "FACE_NOT_OF_PERSON"
	INPUT_IS_NOT_IMAGE_FILE = "INPUT_IS_NOT_IMAGE_FILE" 

class Validation(str, enum.Enum):
	PERSON_ID_NOT_FOUND = "PERSON_ID_NOT_FOUND"
	PERSON_ID_ALREADY_EXIST = "PERSON_ID_ALREADY_EXIST"
	FACE_ID_ALREADY_EXIST = "FACE_ID_ALREADY_EXIST"
	CREATED_FACE = "CREATED_FACE"
	PERSON_ID_HAS_NOT_FACE_ID = "PERSON_ID_HAS_NOT_FACE_ID"

	CONFLICT = "CONFLICT"
	CREATED = "CREATED"
	NOT_ACCEPTABLE = "NOT_ACCEPTABLE"
	UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
	SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"