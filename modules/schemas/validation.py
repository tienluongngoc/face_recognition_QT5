import enum

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