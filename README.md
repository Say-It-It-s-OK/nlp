# 🧠 nlp-server

FastAPI 기반 음성 키오스크용 자연어 처리 서버  
OpenAI GPT-4를 활용해 사용자의 음성 텍스트를 처리하고, Static API 서버로 결과를 전달합니다.

---

## 📁 가상환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 📦 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 🚀 서버 실행

```bash
uvicorn app.main:app --reload
```

> `--reload`: 코드 변경 시 서버 자동 재시작 (개발용)