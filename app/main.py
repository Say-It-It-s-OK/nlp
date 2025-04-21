from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.api import nlp

app = FastAPI()
app.include_router(nlp.router, prefix="/voice")

origins = [
    "http://localhost:3000",  
    "http://localhost:3001",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Say it, it's Okay!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3002)