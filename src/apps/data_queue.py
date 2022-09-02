from src.inferences.base import Singleton
from queue import Queue

class DataQueue(metaclass=Singleton):
    def __init__(self) -> None:
        self.frame_queue = Queue()

    def get_frame_queue(self):
        return self.frame_queue
    
class ResultQueue(metaclass=Singleton):
    def __init__(self) -> None:
        self.result_queue = Queue()

    def get_result_queue(self):
        return self.result_queue