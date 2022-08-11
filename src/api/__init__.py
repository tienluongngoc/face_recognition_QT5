from fastapi import APIRouter
from . import people, faces, embedding_vectors

router = APIRouter()
router.include_router(people.router)
router.include_router(faces.router)
router.include_router(embedding_vectors.router)
