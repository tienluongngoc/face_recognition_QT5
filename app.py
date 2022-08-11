import sys
sys.path.append("./src")
import cv2
from src.apps import FaceRecognitionApp

app =  FaceRecognitionApp()
app.run()