import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import httpx
import asyncio

# .env 로드 (루트에서)
dotenv_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=dotenv_path)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# System Prompt 정의
SYSTEM_PROMPT = """
너는 키오스크 도우미야.
사용자의 요청을 분석해서 다음 JSON 형식으로 intent와 세부 속성을 추출해줘.

가능한 intent: recommend, order, confirm, exit
가능한 categories 값: ["coffee", "drink", "dessert"] (복수 선택 가능)

필터는 다음과 같은 key를 가질 수 있어:
- price: 가격 조건 필터. 예: {"max": 3000}, {"min": 2000, "max": 5000}, 또는 정렬 기준: {"sort": "asc"} (가장 저렴한 순), {"sort": "desc"} (가장 비싼 순)
- caffeine: 카페인 조건. "decaf"로 디카페인 필터링
- tag: 태그 기반 필터. 예:
    - "popular": 인기 많은 메뉴 (유행 포함)
    - "zero": 제로칼로리 및 다이어트용 메뉴
    - "new": 신메뉴
    - "warm": 따뜻한 계열
    - "sweet": 달달한 맛
    - "refresh": 청량한 맛
    - "bitter": 쓴맛 계열
    - "nutty": 고소한/견과류 풍미
    - "creamy": 부드럽고 크리미한 맛
    - "fruity": 과일향 또는 상큼한 맛
    - "gender_male", "gender_female": 성별 기반 추천
    - "young": 10~30대 사용자 요청 시 적용
    - "old": 40대 이상 사용자 요청 시 적용

※ intent는 항상 다음 네 가지 중 하나로만 설정해줘: recommend, order, confirm, exit
※ 사용자가 '커피', '음료', '디저트'와 같은 메뉴 종류를 복수로 언급한 경우, categories 필드를 배열 형태로 설정해주세요.
예: ["coffee", "drink"]
※ 모호한 표현(예: "마실 것")도 의미에 따라 categories로 매핑될 수 있습니다.
※ '제로칼로리' 또는 '다이어트' 요청 시 filters.tag 안에 반드시 "zero"를 포함해주세요.
※ 사용자가 나이대를 언급한 경우, 30대 이하면 "young", 40대 이상은 "old"로 tag에 포함해주세요.
※ "가장 싼", "가장 저렴한", "가장 비싼" 요청 시에는 filters.price.sort 필드를 사용해주세요.
  - 예: {"sort": "asc"} → 저렴한 순 정렬
  - 예: {"sort": "desc"} → 비싼 순 정렬
※ 사용자가 "메뉴 보여줘", "무슨 메뉴 있어?", "커피 메뉴 뭐 있어?" 같은 표현을 사용하면 intent는 반드시 "confirm"으로 설정해주세요.
  - 예: "커피 메뉴 보여줘" → intent: "confirm", categories: ["coffee"]
  - 예: "전체 메뉴 보여줘" → intent: "confirm" (categories 생략 가능)

응답은 항상 아래 JSON 형식으로만 응답해:
{
  "intent": string,
  "categories": array of strings (optional),
  "filters": object (optional),
  "item": object (optional),
  "changes": object (optional),
  "target": string (optional),
  "action": string (optional)
}
"""

def make_messages(user_input: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

def build_backend_payload(intent_result: dict) -> dict:
    intent = intent_result.get("intent")
    request_key = f"query.{intent}" if intent in ["recommend", "confirm"] else f"order.{intent}"

    payload = intent_result.copy()
    payload.pop("intent", None)

    return {
        "request": request_key,
        "payload": payload
    }

async def send_to_backend(intent_result: dict):
    data = build_backend_payload(intent_result)
    async with httpx.AsyncClient() as client:
        await client.post("http://localhost:3000/api/handle", json=data)

def call_openai(messages: List[Dict[str, str]]) -> Dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        # 🔧 테스트 중이므로 백엔드 전송은 주석 처리 (완료 후 주석 해제)
        # asyncio.run(send_to_backend(parsed))

        return parsed
    except Exception as e:
        return {"error": str(e)}
    
async def handle_text(text: str):
    messages = make_messages(text)
    result = call_openai(messages)
    return result
    

# 테스트용 실행
if __name__ == "__main__":
    user_input = "살 안찌는 마실 거 뭐 있어?"
    messages = make_messages(user_input)
    result = call_openai(messages)
    print("GPT 응답 결과:", result)
