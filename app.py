import sys
sys.path.append("./src")
import cv2
from src.apps import person_management_instance, face_management_instance

# app =  FaceRecognitionApp()
# app.run()

# person_management_instance.insert_person("tienln", "Luong Ngoc Tien")
# person_management_instance.delete_person_by_id("tienln")
# people = person_management_instance.select_all_people(0,10)
# person = person_management_instance.select_person_by_id("tienln")


# img = cv2.imread("images/cr7.jpg")
# face_management_instance.insert_face("tienln","125d3s4", img)
# print(face_management_instance.select_all_face_of_person("tienln", 0, 10))
face_management_instance.delete_face_by_id("tienln", "125d34")