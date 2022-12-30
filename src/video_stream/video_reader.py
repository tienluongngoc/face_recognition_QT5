import queue
import time
import threading
import numpy as np
from queue import Queue
from threading import Thread
from .observer import Subject, Subscriber
import cv2
import time

class VideoReader(Subject, Thread):

    def __init__(self, config, frame_queue: Queue):
        Thread.__init__(self)
        super(VideoReader, self).__init__()
        self.camera_url = 0
        self.sleep_time = 0#config[""]
        # self.camera_id = config["camera_id"]
        # self.location = config["location"]
        # self.service_name = config["service_name"]

        self.frame_queue = frame_queue
        self.frame = np.zeros((2, 2))
        self.is_stop = False
        self.pause = False
        self.cap = cv2.VideoCapture(self.camera_url)
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.frame_count = 0
        
    def stop_thread(self):
        self.is_stop = True

    def run(self):
        while not self.is_stop:
            if not self.pause:
                try:
                    grabbed, self.frame = self.cap.read()
                    if grabbed == True:
                        time.sleep(self.sleep_time)
                        frame_data = {"image": self.frame,
                                    #   "location": self.location,
                                    #   "camera_id": self.camera_id, 
                                    #   "service_name": self.service_name
                                    }
                        self.frame_queue.put(frame_data)
                        self.notify()
                        
                    if self.frame_queue.qsize() >= 2:
                        self.frame_queue.get()
                    self.new_frame_time = time.time()
                    self.prev_frame_time = self.new_frame_time
                except Exception as ex:
                    self.log_error(ex)

class Viewer(Subscriber, Thread):
    def __init__(self):
        Thread.__init__(self)
        super(Viewer, self).__init__()
        self.frame = np.zeros((1, 1))
        self.is_stop = False

    def update(self, subject):
        self.frame = subject.frame
        self.is_stop = subject.is_stop

    def get_frame(self):
        return self.frame

    def stop_thread(self):
        self.is_stop = True

    def show(self):
        while not self.is_stop:
            image = self.frame
            image = cv2.resize(image, (1280, 1024))
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_stop = True
                break
        cv2.destroyAllWindows()

