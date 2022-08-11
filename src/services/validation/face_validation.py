from typing import List
from schemas.validation import ImageValidation
from models import PersonDoc
import cv2
import numpy as np
from inferences import det_infer, enc_infer, reg_infer
from inferences.utils.face_detect import Face
import os, glob

class FaceValidation:
	def __init__(self, face_config):
		self.face_config = face_config
		self.engine = enc_infer.config.engine_default

	def encode(self, image: np.ndarray) -> np.ndarray:
		detection_results = det_infer.detect(image)
		if detection_results[0].shape[0] == 0 or detection_results[1].shape[0] == 0:
			return np.array([])
		detection_results = self.get_largest_bbox(detection_results)
		face = Face(
			bbox=detection_results[0][:4], 
			kps=detection_results[1][0], 
			det_score=detection_results[0][-1]
		)
		encode_results = enc_infer.get(image, face)
		return encode_results

	def check_face_liveness(self, image):
		detection_results = det_infer.detect(image)
		if detection_results[0].shape[0] == 0 or detection_results[1].shape[0] == 0:
			return None
		detection_results = self.get_largest_bbox(detection_results)
		bbox = detection_results[0][:4]
		# live = anti_infer.check(image, bbox)
		live = True
		return live


	def check_image(self, image: np.ndarray, person_doc: PersonDoc) -> ImageValidation:
		if image is None:
			return ImageValidation.INPUT_IS_NOT_IMAGE_FILE
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# check if image has already in database
		img_path_root = self.face_config["path"]
		person_img_path = os.path.join(img_path_root, person_doc["id"])
		if os.path.exists(person_img_path):
			end_image = self.face_config["end"]
			current_image_paths = glob.glob(person_img_path + f"/*{end_image}")
			if len(current_image_paths) != 0:
				current_images = [cv2.imread(image_path) for image_path in current_image_paths]
				current_images = [cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) for current_image in current_images]
				for current_image in current_images:
					if np.array((image == current_image)).all():
						return ImageValidation.IMAGE_HAS_ALREADY_IN_DB
		
		return ImageValidation.IMAGE_IS_VALID

	def get_largest_bbox(self, detection_results: List[np.ndarray]) -> List[np.ndarray]:
			bboxes = detection_results[0]
			kpss = detection_results[1]
			largest_bbox = max(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
			kps_corr = kpss[np.where((bboxes==largest_bbox).all(axis=1))]
			detection_largest = [largest_bbox, kps_corr]
			return detection_largest

	def validate_face(self, image: np.ndarray, person_doc: PersonDoc) -> ImageValidation:
		# check format and unique of image
		check_result = self.check_image(image, person_doc)
		if check_result != ImageValidation.IMAGE_IS_VALID:
			return check_result
				
		# check if image has no face
		detection_results = det_infer.detect(image)
		num_faces = len(detection_results[0])
		if num_faces == 0:
			return ImageValidation.IMAGE_HAS_NO_FACE
		else:
			detection_results = self.get_largest_bbox(detection_results)
		
		# encode face in image
		face_encoded = self.encode(image)

		# check when this person has no any faces in database
		has_face = True
		if ("faces" not in person_doc.keys()) or person_doc["faces"] is None:
			has_face = False
		elif len(person_doc["faces"]) == 0:
			has_face = False
		if not has_face:
			person_info = reg_infer.search(face_encoded)
			if len(person_info) != 0:
				if person_info["person_id"] != "continue":
					if person_info["person_id"] == "unrecognize":
						return ImageValidation.IMAGE_IS_VALID
					elif person_info["person_id"] == person_doc["id"]:
						return ImageValidation.IMAGE_IS_VALID
					else:
						return ImageValidation.FACE_NOT_OF_PERSON
		else:
			# check if face in image is this person's
			current_embedding_vectors = []
			for face in person_doc["faces"]:
				if "vectors" not in face.keys():
					continue
				elif len(face["vectors"]) is None:
					continue
				elif len(face["vectors"]) == 0:
					continue
				current_embedding_vectors += [
					v["value"] for v in face["vectors"] if v["engine"]==self.engine
				]
			if len(current_embedding_vectors) == 0:
				return ImageValidation.IMAGE_IS_VALID
			else:
				current_embedding_vectors = np.array(current_embedding_vectors, dtype=np.float32)
				sim_values = [enc_infer.is_same(
					np.array(x, dtype=np.float32), np.array(face_encoded, dtype=np.float32)
					) for x in current_embedding_vectors]
				if np.array(sim_values).all():
					return ImageValidation.IMAGE_IS_VALID
				else:
					return ImageValidation.FACE_NOT_OF_PERSON
		


		

		