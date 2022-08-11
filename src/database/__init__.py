from .PersonDB import PersonDatabase
from configs import mongodb_config

PersonDB_instance = PersonDatabase(config=mongodb_config)