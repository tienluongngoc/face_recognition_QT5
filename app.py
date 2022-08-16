from math import frexp
import re
import sys
sys.path.append("./src")
import cv2
from src.apps import person_management_instance, face_management_instance
from src.apps.face_recognition import FaceRecognition

app =  FaceRecognition()
app.run()

# person_management_instance.insert_person("cr7", "Ronaldo")
# people = person_management_instance.select_all_people(0,10)
# print(people)
# person_management_instance.delete_person_by_id("tienln")
# people = person_management_instance.select_all_people(0,10)
# person = person_management_instance.select_person_by_id("tienln")
# person_management_instance.delete_all_people()


# img = cv2.imread("cr7.jpg")
# result = face_management_instance.insert_face("cr7","02", img)
# print(result)
# print(face_management_instance.select_all_face_of_person("tienln", 0, 10))
# res = face_management_instance.delete_face_by_id("tienln", "125d34")
# res = face_management_instance.delete_all_face("tienlnd")
# print(res)