from fastapi import APIRouter, status, Response, HTTPException
from fastapi import UploadFile, File

router = APIRouter(prefix='/embedding_vectors', tags=['embedding_vectors'])