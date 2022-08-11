from pydantic import BaseModel, Field
from typing import List


class EmbeddingVectorDoc(BaseModel):
	id: str = Field(None)
	engine: str = Field(None)
	value: List[float] = Field([])

	def __init__(self, id: str, engine: str, value: list = None) -> None:
		super(EmbeddingVectorDoc, self).__init__()
		self.id = id
		self.engine = engine
		self.value = value

	class Config:
		schema_extra = {
			"example": {
				"id": "svasv",
				"engine": "ArcFace",
				"value": [1, 2, 3]
			}
		}

class FaceDoc(BaseModel):
	id: str = Field(None)
	imgPath: str = Field(None)
	vectors: List[EmbeddingVectorDoc] = Field([])

	def __init__(self, id: str = "", imgPath: str = "", vectors: List[EmbeddingVectorDoc] = None) -> None:
			super(FaceDoc, self).__init__()
			self.id = id
			self.imgPath = imgPath
			self.vectors = vectors

	class Config:
		schema_extra = {
			"example": {
				"id": "sdasv",
				# "imgPath": "/src/hoangnv",
			}
		}

class PersonDoc(BaseModel):
	id: str = Field(None)
	name: str = Field(None)
	faces: List[FaceDoc] = Field([])

	def __init__(self, id: str, name: str, faces: List[FaceDoc] = None) -> None:
		super(PersonDoc, self).__init__()
		self.id = id
		self.name = name
		self.faces = faces

	class Config:
		schema_extra = {
			"example": {
				"id": "unique",
				"name": "hoangnv",
			}
		}
