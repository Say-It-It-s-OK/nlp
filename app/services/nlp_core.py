import os
import openai
import httpx
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

async def handle_text(text: str) -> str:
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 키오스크 음성 주문을 도와주는 어시스턴트입니다."},
                {"role": "user", "content": text}
            ]
        )
        reply = response.choices[0].message.content

        # Static API 서버로 결과 전송
        static_api_url = os.getenv("STATIC_API_URL")
        async with httpx.AsyncClient() as client:
            await client.post(static_api_url, json={"intent_result": reply})

        return reply
    except Exception as e:
        return f"Error: {e}"