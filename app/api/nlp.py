from fastapi import APIRouter, Request
from app.services import nlp_core

router = APIRouter()

@router.post("/process")
async def process_command(request: Request):
    body = await request.json()
    text = body.get("text", "")
    result = await nlp_core.handle_text(text)
    return {"response": result}