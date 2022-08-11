import sys
sys.path.append("./src")
from inferences import det_infer, enc_infer
from inferences.utils.face_detect import Face
import cv2
import numpy as np
import glob
from uvicorn.config import logger


def compute_sim(feat1, feat2):
	from numpy.linalg import norm
	feat1 = feat1.ravel()
	feat2 = feat2.ravel()
	sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
	return sim

def recognize(rootpath: str, threshold: float) -> None:
	imgpaths = glob.glob(rootpath+"/*.jpg")
	face_encodes = []
	for imgpath in imgpaths:
		img = cv2.imread(imgpath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = det_infer.detect(img)
		face = Face(
			bbox=results[0][:4], 
			kps=results[1][0], 
			det_score=results[0][-1]
		)
		encode_result = enc_infer.get(img, face)
		face_encodes.append(encode_result)

	in_img = face_encodes[0]
	db = face_encodes[1:]

	sim_values = [compute_sim(in_img, img) for img in db]

	for i, value in enumerate(sim_values):
		if value > threshold:
			logger.info(f"{imgpaths[0]} vs {imgpaths[i+1]} is difference person's face")
		else:
			logger.info(f"{imgpaths[0]} vs {imgpaths[i+1]} is a same person's face")

if __name__ == "__main__":
	rootpath = "./test/images"
	threshold = 0.6
	recognize(rootpath, threshold)
