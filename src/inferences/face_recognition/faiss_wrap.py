import faiss
import numpy as np
from database import PersonDatabase
import threading, queue, time, enum
from datetime import datetime
import os
from ..base import Singleton
from configs import FaissConfig
from uvicorn.config import logger

class ChangeEvent(enum.Enum):
	remove_person = 1
	remove_vector = 2
	update_name = 3
	update_id = 4
	add_vector = 5
	update_vector = 6
	remove_all_db = 7

class FAISS(metaclass=Singleton):
	def __init__(self, config: FaissConfig, local_db: PersonDatabase):
		self.config = config
		self.index = faiss.IndexFlatIP(config.dim)
		assert config.device in ["gpu", "cpu"], f"[{datetime.now()}][{self.__class__.__name__}]: Error device, device is only one \
					of two values: ['cpu', 'gpu']"
		self.local_db = local_db
		self.is_trained = False
		self.change_db_events = queue.Queue()
		self.db_change = False
		self.change_db_worker = threading.Thread(target=self.run_change_db)
		self.reload_model_worker = threading.Thread(target=self.run_reload_model)
		self.change_all_db_worker = threading.Thread(target=self.run_change_all_db)
		self.initialize()

	def initialize(self):
		self.local_db.initialize_local_db()
		if not os.path.exists(self.config.model_path):
			self.train()
			logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Write model to {self.config.model_path}.")
		else:
			self.index = faiss.read_index(self.config.model_path)
			self.is_trained = True
			logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Read model from {self.config.model_path}.")
		if self.config.device == "gpu":
			res = faiss.StandardGpuResources()
			self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
		self.change_db_worker.start()
		self.reload_model_worker.start()
		self.change_all_db_worker.start()
	
	def stop(self):
		self.change_db_worker.join()
		self.reload_model_worker.join()
		self.change_all_db_worker.join()

	def run_change_db(self):
		while True:
			while not self.change_db_events.empty():
				event = self.change_db_events.get()
				key = list(event.keys())[0]
				if key == ChangeEvent.remove_person:
					self.local_db.remove_person_from_local_db(*event[key])
				elif key == ChangeEvent.remove_vector:
					self.local_db.remove_vector_from_local_db(*event[key])
				elif key == ChangeEvent.update_name:
					self.local_db.update_name_to_local_db(*event[key])
				elif key == ChangeEvent.update_id:
					self.local_db.update_id_to_local_db(*event[key])
				elif key == ChangeEvent.add_vector:
					self.local_db.add_vector_to_local_db(*event[key])
				elif key == ChangeEvent.update_vector:
					self.local_db.update_vector(*event[key])
				elif key == ChangeEvent.remove_all_db:
					self.local_db.remove_all_db()

				self.db_change = True
				logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Local Person Database are changed.")
			time.sleep(self.config.solve_event_delay)
	
	def run_reload_model(self):
		while True:
			if self.db_change:
				self.reload()
				self.db_change = False
				logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Recognition model are retrained.")
			time.sleep(self.config.retrain_delay)

	def run_change_all_db(self):
		while True:
			time.sleep(self.config.reload_all_db_delay)
			logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Reload all document for recognition model.")
			self.local_db.initialize_local_db()
			self.db_change = True

	def add_change_event(self, event: ChangeEvent, params: list):
		self.change_db_events.put({event: params})

	def train(self):
		if (len(self.local_db.vectors.keys())) == 0:
			self.is_trained = False
			return
		vectors = np.array(
			[vector for vector in self.local_db.vectors.values()], 
			dtype=np.float32
		)
		self.index.add(vectors)
		self.is_trained = self.index.is_trained
		if os.path.exists(self.config.model_path):
			os.remove(self.config.model_path)
		faiss.write_index(self.index, self.config.model_path)
	
	def search(self, embedding_vectors: np.ndarray, nearest_neighgbors: int = 5):
		# check if model is trained completely before searching
		if not (self.is_trained or self.index.is_trained):
			return {}
		# search nearest embedding vectors
		D, I = self.index.search(np.array([embedding_vectors]), nearest_neighgbors) 
		person_indexes = []
		# select embedding vectors with cosin > threshold
		for i, dis in enumerate(D[0]):
			if dis > self.config.threshold:
				person_indexes.append(I[0][i])
		if len(person_indexes) == 0:
			return {"person_id": "unrecognize"}  
				
		if len(list(self.local_db.vectors.keys())) != 0:
			person_infos = [self.local_db.people[list(self.local_db.vectors.keys())[x]] for x in person_indexes]
		else:
			return {"person_id": "unrecognize"}  
		# check if all results are only one person's
		person_ids = [x["person_id"] for x in person_infos]
		if len(set(person_ids)) == 1:
			return person_infos[0]
		else:
			# return server error
			return {"person_id": "continue"}               
	
	def reload(self):
		if len(self.local_db.vectors.keys()) == 0 or len(self.local_db.people.keys()) == 0:
			self.is_trained = False
			return
		vectors = np.array([vector for vector in self.local_db.vectors.values()], dtype=np.float32)
		self.index.reset()
		self.index.add(vectors)
		self.is_trained = self.index.is_trained
		if os.path.exists(self.config.model_path):
			os.remove(self.config.model_path)
		if self.config.device == "gpu":
			cpu_index = faiss.index_gpu_to_cpu(self.index)
			faiss.write_index(cpu_index, self.config.model_path)
		else:
			faiss.write_index(self.index, self.config.model_path)
		logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Rewrite model to {self.config.model_path}.")
