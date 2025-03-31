from fastapi import FastAPI
from app.api import nlp

app = FastAPI()
app.include_router(nlp.router, prefix="/voice")
