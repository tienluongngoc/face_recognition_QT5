import sys
sys.path.append("./src")
import cv2
from src.apps import FaceRecognitionApp

np_image =  cv2.imread("images/cr72.jpg")
app =  FaceRecognitionApp()
results = app.recognize(np_image)
print(results)
