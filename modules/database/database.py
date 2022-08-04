
import sys
import yaml
from pymongo import MongoClient
from modules.utils.utils import SingletonType
from modules.utils.logger import Logger

class Database(object,metaclass=SingletonType):
    
    def __init__(self, config=None):
        self.logger = Logger.__call__().get_logger()

        with open("configs/config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        try:
            database_config = config["database"]
            self.host = database_config["host"]
            self.port = database_config["port"]
            self.username = database_config["user_name"]
            self.password = database_config["password"]
            self.database_name = database_config["database_name"]

            connection_str = "mongodb://{}:{}@{}:{}/".format(self.username,self.password, self.host, self.port)
            connection = MongoClient(connection_str)
            database_list = connection.list_database_names()
            
            self.logger.info("Connect to {} with database {}".format(connection_str, self.database_name))

            if self.database_name  not in database_list:
                self.logger.warning("{} database is not exists...".format(self.database_name))
            self.database = connection[self.database_name]
        except Exception as e:
            self.logger.error(str(e))

    def get_connection(self):
        return self.database


     
