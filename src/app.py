from fastapi import FastAPI
import api

app = FastAPI(
	title='Accountr',
	version='1.0.0'
)

app.include_router(api.router)