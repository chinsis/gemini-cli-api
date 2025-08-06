#!/usr/bin/env python3
"""
Gemini CLI API 包装服务器
集成 OAuth2 密码模式 + JWT 鉴权，Token 永不过期
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import subprocess
import uuid
import datetime
import logging
import os
from typing import Optional, List
from contextlib import asynccontextmanager
import uvicorn

from jose import JWTError, jwt
from passlib.context import CryptContext

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', '')

# ----------- OAuth2 + JWT -------------------

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "4BM29fYLC7sVkc9aLUEfC3yWaPHgD3hJ")
PASSWORD = os.environ.get("PASSWORD", "R27Qwn68nP7gaaS3")
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

fake_users_db = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Example",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash(PASSWORD),
        "disabled": False,
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str) -> Optional[UserInDB]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db, username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法认证的凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="用户已禁用")
    return current_user

# ----------- FastAPI 启动 -------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEFAULT_PROJECT_ID:
        logger.info(f"✅ 默认Google Cloud项目: {DEFAULT_PROJECT_ID}")
    else:
        logger.warning("⚠️  未设置默认GOOGLE_CLOUD_PROJECT")
    yield
    logger.info("🔻 Gemini CLI API 服务器关闭")

app = FastAPI(
    title="Gemini CLI API",
    description="包装Gemini CLI的简单API服务，集成OAuth2密码模式 + JWT鉴权",
    lifespan=lifespan
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "gemini-2.5-pro"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    project_id: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

class SimpleChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemini-2.5-pro"
    project_id: Optional[str] = None

class SimpleChatResponse(BaseModel):
    response: str
    status: str
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Gemini CLI API 服务器运行中", "docs": "/docs"}

@app.get("/health")
async def health_check():
    try:
        result = subprocess.run(["gemini", "--help"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return {"status": "healthy", "gemini_cli": "available"}
        return {"status": "unhealthy", "gemini_cli": "not available"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

def execute_gemini_command(prompt: str, model: str = "gemini-2.5-pro", project_id: str = None) -> tuple[str, str, int]:
    try:
        current_project = project_id or DEFAULT_PROJECT_ID
        if not current_project:
            return "", "错误：需要指定project_id", 1
        env = dict(os.environ)
        env.update({
            'GOOGLE_CLOUD_PROJECT': current_project,
            'TERM': 'xterm-256color',
            'HOME': os.path.expanduser('~'),
        })
        shell_command = f'echo "" | gemini -m "{model}" -p "{prompt}"'
        result = subprocess.run(shell_command, shell=True, capture_output=True, text=True, timeout=60, env=env)
        if result.returncode == 0:
            return result.stdout.strip(), result.stderr, result.returncode
        return "", result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, current_user: User = Depends(get_current_active_user)):
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    prompt = user_messages[-1].content
    output, error, return_code = execute_gemini_command(prompt, request.model, request.project_id)
    if return_code != 0:
        raise HTTPException(status_code=500, detail=f"Gemini CLI error: {error}")
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(datetime.datetime.now().timestamp()),
        "model": "gemini-cli-proxy",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output},
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }

@app.post("/chat", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest, current_user: User = Depends(get_current_active_user)):
    output, error, return_code = execute_gemini_command(request.message, request.model, request.project_id)
    if return_code == 0:
        return SimpleChatResponse(response=output, status="success")
    return SimpleChatResponse(response="", status="error", error=f"Gemini CLI 错误: {error}")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    print("🚀 启动 Gemini CLI API 服务器（带OAuth2/JWT认证）...")
    print("📖 API 文档: http://localhost:8000/docs")
    print("🔗 健康检查: http://localhost:8000/health")
    print("🔑 获取Token接口: http://localhost:8000/token")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
