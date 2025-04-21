from fastapi import APIRouter, Request
from app.services import openai_client

router = APIRouter()

@router.post("/process")
async def process_command(request: Request):
    body = await request.json()
    text = body.get("text", "")
    result = await openai_client.handle_text(text)
    return {"response": result}
