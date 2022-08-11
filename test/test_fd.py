import sys
sys.path.append("./src")
from inferences import det_infer
import cv2
import glob

def visualize(img, bboxes, kpss):
	for bbox, kps in zip(bboxes, kpss):
		left, top, right, bottom, score = bbox
		left, top, right, bottom = int(left), int(top), int(right), int(bottom)
		cv2.rectangle(img, (left, top), (right, bottom), color=(0, 255, 0), thickness=3)
		cv2.putText(
			img, "%.2f" % (score), (left - 10, top - 10),
			color=(0, 0, 255), thickness=3, 
			fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale= 5	
		)
		for kp in kps:
			kp = (int(kp[0]), int(kp[1]))
			cv2.circle(img, kp, 3, color=(255, 0, 0), thickness=3)

def detect(rootpath: str) -> None:
	imgpaths = glob.glob(rootpath+"/*.jpg")
	for imgpath in imgpaths:
		img = cv2.imread(imgpath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		bboxes, kpss = det_infer.detect(img)
		img_cp = img.copy()
		visualize(img_cp, bboxes, kpss)
		save_path = imgpath[:-4]+"_det_local.jpeg"
		img_cp = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
		cv2.imwrite(save_path, img_cp)


if __name__ == "__main__":
	rootpath = "./test/images"
	detect(rootpath)
