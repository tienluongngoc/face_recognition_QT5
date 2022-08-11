import pymongo
from datetime import datetime
from uvicorn.config import logger

class BaseDatabase(object):
	def __init__(self, config):
		self.hostname = config["hostname"]
		self.port = config["port"]
		self.user = config["user"]
		self.password = config["password"]
		if (self.user == None or self.password == None) or ((self.user == "" or self.password == "")):
			self.url = f"mongodb://{self.hostname}:{self.port}"
		else:
			self.url = f"mongodb://{self.user}:{self.password}@{self.hostname}:{self.port}"
		self.maxSevSelDelay = config["maxSevSelDelay"]
		self.Initialize()

	def Initialize(self):
		try:
			self.client = pymongo.MongoClient(
				self.url,
				serverSelectionTimeoutMS=self.maxSevSelDelay
			)
			logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Connected to MongoDB.")
		except pymongo.errors.ServerSelectionTimeoutError as err:
			logger.info(f"[{datetime.now()}][{self.__class__.__name__}]: Cannot connect to MongoDB: {err}")
