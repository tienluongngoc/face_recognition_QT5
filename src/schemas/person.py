from pydantic import BaseModel, Field

class SimplePerson(BaseModel):
	id: str = Field(None)
	name: str = Field(None)

	def __init__(self, id: str, name: str) -> None:
			super(SimplePerson, self).__init__()
			self.id = id
			self.name = name
	
	class Config:
		schema_extra = {
			"example": {
				"id": "dfasdfas",
				"name": "hoangnv"
			}
		}

class Face(BaseModel):
	pass

class Embedding_vector(BaseModel):
	pass