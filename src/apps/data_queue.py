from inferences import Singleton
from queue import Queue

class DataQueue(metaclass=Singleton):
    def __init__(self) -> None:
        self.frame_queue = Queue()

    def get_frame_queue(self):
        return self.frame_queue