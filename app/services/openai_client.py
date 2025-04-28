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

SYSTEM_PROMPT = """
너는 키오스크 도우미야.
사용자의 요청을 분석해서 다음 JSON 형식으로 intent와 세부 속성을 추출해줘.

가능한 intent: recommend, order, confirm, exit, error
가능한 categories 값: ["커피", "음료", "디저트", "디카페인"] (복수 선택 가능)

필터는 다음과 같은 key를 가질 수 있어:
- price: 가격 조건 필터. 예: {"max": 3000}, {"min": 2000, "max": 5000}, {"sort": "asc"}, {"sort": "desc"}
- tag: 메뉴의 맛이나 특징을 나타내는 필터. 예:
    - "popular", "zero", "new", "warm", "cold", "refresh", "sweet", "bitter", "nutty", "creamy", "fruity", "gender_male", "gender_female", "young", "old"
- caffeine: 카페인 조건 필터. "decaf" (디카페인 요청 시 사용)
- include_ingredients: 반드시 포함해야 하는 재료 리스트. 예: {"include_ingredients": ["딸기"]}
- exclude_ingredients: 반드시 제외해야 하는 재료 리스트. 예: {"exclude_ingredients": ["우유"]}

item 안에 들어갈 수 있는 정보:
- name: 메뉴 이름
- size: "S", "M", "L" 중 하나
- shot: "extra" (샷 추가) 또는 "none" (샷 제거)

※ intent는 항상 recommend, order, confirm, exit, error 중 하나로 설정해야 합니다.

※ 사용자가 '커피', '음료', '디저트', '디카페인'을 복수로 언급한 경우, categories를 배열 형태로 설정합니다.

※ 사용자가 '마실 것', '마실 거'처럼 음료 전반을 지칭하는 경우, categories는 ["커피", "음료", "디카페인"]를 모두 포함합니다.

※ 사용자가 '전체 메뉴', '다 보여줘', '모든 메뉴'를 요청하는 경우, intent는 confirm으로 설정하고 categories는 생략합니다.

※ 사용자가 '시원한', '아이스'를 말하면 filters.tag에 "cold"를 추가하고, '따뜻한', '추운'을 말하면 "warm"을 추가합니다.

※ 사용자가 '청량한', '톡 쏘는'을 말하면 filters.tag에 "refresh"를 추가합니다.

※ 사용자가 '제로칼로리', '다이어트'를 요청하면 filters.tag에 "zero"를 반드시 포함합니다.

※ 사용자가 '가장 싼', '가장 비싼' 메뉴를 요청하면 filters.price.sort를 사용합니다.

※ 사용자가 특정 재료를 포함하거나 제외해서 요청하면 filters.include_ingredients 또는 filters.exclude_ingredients를 사용합니다.

※ 사용자가 카페인 없는 음료를 요청하면 filters.caffeine을 "decaf"로 설정하고, 필요에 따라 categories에 "디카페인"을 추가합니다.

※ 메뉴 추천(intent: recommend) 중 사용자가 사이즈 요청을 하면 item.size를 추가해 "S", "M", "L" 중 하나로 설정합니다.

※ 메뉴 주문(intent: order.add, order.update) 중에는 item.name과 함께 필요한 옵션(size, shot 등)을 함께 설정해야 합니다.

※ 사용자가 사이즈(size), 온도(tag: warm/cold), 샷 추가(extra shot)와 관련된 변경 요청을 할 경우, 메뉴명이 명시되지 않더라도 intent를 order.update로 설정하고 관련 필드를 채워주세요. error로 처리하지 않습니다.

※ 사용자가 추천 결과에 대해 "싫어", "다른 거", "좋아"와 같이 단답형으로 응답할 경우 현재 대화 흐름에 따라 다음과 같이 처리합니다:
- recommend 흐름 중: "싫어", "다른 거" → action: "reject" 또는 "retry"
- confirm 흐름 중(장바구니, 주문 내역 확인 등): "맞아" → action: "confirm", "아니야" → action: "modify"
- order 흐름 중: "수정할래", "변경할래" → action: "update", "취소할래" → intent: "exit"
- pay 흐름 중: "결제할게", "진행할게" → intent: "order.pay", "취소할래" → intent: "exit"

※ 사용자의 요청이 키오스크 주문과 직접 관련이 없는 경우(예: 날씨 묻기, 가게 위치 질문 등)에는 intent를 error로 설정합니다.

응답은 항상 다음 JSON 형식으로 해주세요:
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
    request_key = f"query.{intent}" if intent in ["recommend", "confirm", "error"] else f"order.{intent}"

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
    user_input = "다른거"
    messages = make_messages(user_input)
    result = call_openai(messages)
    print("GPT 응답 결과:", result)
