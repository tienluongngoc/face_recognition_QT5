from fastapi import APIRouter, Response, status
from models.person import PersonDoc
from schemas import SimplePerson
from typing import List
from x import PersonCRUD_instance
from urllib.parse import unquote

router = APIRouter(prefix='/people', tags=['people'])

@router.get(
	"", 
	response_model=List[PersonDoc],
	status_code=status.HTTP_200_OK
)
async def get_all_people(skip: int = 0, limit: int = 10) -> list:
	PeopleDocs = PersonCRUD_instance.select_all_people(skip, limit)
	return PeopleDocs

@router.get(
	"/{person_id}",
	response_model= PersonDoc,
	status_code=status.HTTP_200_OK
)
async def select_person_by_ID(person_id: str):
	person_id = unquote(person_id)
	person_doc = PersonCRUD_instance.select_person_by_id(person_id)
	return person_doc

@router.post(
	"",
	response_model= PersonDoc,
	status_code=status.HTTP_201_CREATED	
)
async def insert_one_person(id: str, name: str):
	id,name = unquote(id), unquote(name)
	simple_person = SimplePerson(id=id, name=name)
	person_doc = PersonCRUD_instance.insert_person(person=simple_person)
	return person_doc

@router.post(
	"/{person_id}/name",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
async def update_name(person_id: str, name: str):
	person_id, name = unquote(person_id), unquote(name)
	PersonCRUD_instance.update_person_name(person_id, name)

@router.post(
	"/{person_id}/id",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
async def update_id(person_id: str, new_id: str):
	person_id, new_id = unquote(person_id), unquote(new_id)
	PersonCRUD_instance.update_person_id(person_id, new_id)

@router.delete(
	"/{id}",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT	
)
async def delete_person_by_ID(id: str):
	PersonCRUD_instance.delete_person_by_id(id)


@router.delete(
	"",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
async def delete_all_people():
	PersonCRUD_instance.delete_all_people()
