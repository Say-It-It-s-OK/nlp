import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

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
가능한 category: coffee, drink, dessert

필터는 다음과 같은 key를 가질 수 있어:
- price: 가격 조건 필터. 예: {"max": 3000}, {"min": 2000, "max": 5000}
- caffeine: 카페인 조건. "decaf"로 디카페인 필터링
- tag: 인기 메뉴, 제로칼로리 등 분류 태그. 예: ["popular", "zero", "20s"]
- ingredient_exclude: 제외할 성분 필터. 예: ["우유", "견과류"]

반드시 아래 형식으로만 응답해:
{
  "intent": string,
  "category": string (optional),
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

def call_openai(messages: List[Dict[str, str]]) -> Dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

# 테스트용 실행
if __name__ == "__main__":
    user_input = "20대가 좋아하는 음료 추천해줘"
    messages = make_messages(user_input)
    result = call_openai(messages)
    print("GPT 응답 결과:", result)
