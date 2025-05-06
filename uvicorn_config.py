import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("SERVER_PORT", 3002))

if __name__ == "__main__":
   uvicorn.run("app.main:app", host="0.0.0.0", port=3002, reload=True)
