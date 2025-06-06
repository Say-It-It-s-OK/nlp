from fastapi import APIRouter, Request
from app.services import openai_client

router = APIRouter()

@router.post("/process")
async def process_command(request: Request):
    body = await request.json()
    text = body.get("text", "")
    session_id = body.get("sessionId", "")
    page = body.get("page", "")
    result = await openai_client.handle_text(text, session_id, page)
    return {"response": result}
