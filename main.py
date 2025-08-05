#!/usr/bin/env python3
"""
Gemini CLI API 包装服务器
一个简单的API服务器，用来包装Gemini CLI调用，集成OAuth2密码模式 + JWT鉴权
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

from jose import JWTError, jwt  # 新增：JWT相关
from passlib.context import CryptContext  # 新增：密码哈希

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置优先级设计：请求参数 > 环境变量 > 错误
DEFAULT_PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', '')

# ----------- OAuth2 + JWT 配置开始 -------------------

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "4BM29fYLC7sVkc9aLUEfC3yWaPHgD3hJ")
PWD = os.environ.get("PWD", "R27Qwn68nP7gaaS3")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # token有效期1小时

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 模拟用户数据库，生产请换成真正数据库
fake_users_db = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Example",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash(PWD),  # 明文密码是 secret
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
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + (expires_delta or datetime.timedelta(minutes=15))
    to_encode.update({"exp": expire})
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

# ----------- OAuth2 + JWT 配置结束 -------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    if DEFAULT_PROJECT_ID:
        logger.info(f"✅ 默认Google Cloud项目: {DEFAULT_PROJECT_ID}")
        logger.info("💡 可在请求中使用 project_id 字段覆盖默认值")
    else:
        logger.warning("⚠️  未设置默认GOOGLE_CLOUD_PROJECT")
        logger.info("💡 请在每个请求中传递 project_id，或设置环境变量")
    
    yield
    
    logger.info("🔻 Gemini CLI API 服务器关闭")

app = FastAPI(
    title="Gemini CLI API",
    description="包装Gemini CLI的简单API服务，集成OAuth2密码模式 + JWT鉴权",
    lifespan=lifespan
)

# OpenAI兼容的数据模型
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "gemini-2.5-pro"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    project_id: Optional[str] = None  # 可选的项目ID

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

class SimpleChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemini-2.5-pro"
    project_id: Optional[str] = None  # 可选的项目ID

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
        result = subprocess.run(
            ["gemini", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return {"status": "healthy", "gemini_cli": "available"}
        else:
            return {"status": "unhealthy", "gemini_cli": "not available"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/config/status")
async def get_config_status():
    return {
        "default_project_id": DEFAULT_PROJECT_ID,
        "has_default": bool(DEFAULT_PROJECT_ID),
        "design_philosophy": "Stateless API - project_id per request",
        "usage": {
            "with_default": "project_id字段可选，不传则使用环境变量",
            "without_default": "project_id字段必填",
            "override": "请求中的project_id优先级最高"
        },
        "example_request": {
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "project_id": "optional-project-id"
        }
    }

def execute_gemini_command(prompt: str, model: str = "gemini-2.5-pro", project_id: str = None) -> tuple[str, str, int]:
    try:
        current_project = project_id or DEFAULT_PROJECT_ID
        
        if not current_project:
            return "", "错误：需要指定project_id。请在请求中传递project_id或设置GOOGLE_CLOUD_PROJECT环境变量", 1
        
        env = dict(os.environ)
        env.update({
            'GOOGLE_CLOUD_PROJECT': current_project,
            'TERM': 'xterm-256color',
            'HOME': os.path.expanduser('~'),
        })
        
        logger.info(f"Executing gemini CLI with project: {current_project} (source: {'request' if project_id else 'default'})")
        
        shell_command = f'echo "" | gemini -m "{model}" -p "{prompt}"'
        
        result = subprocess.run(
            shell_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=os.path.expanduser('~')
        )
        
        if result.returncode == 0:
            logger.info("Gemini CLI executed successfully.")
            return result.stdout.strip(), result.stderr, result.returncode
        else:
            logger.error(f"Gemini CLI failed: {result.stderr}")
            return "", result.stderr, result.returncode
        
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1

# ----------- 新增 OAuth2 令牌获取接口 -------------------

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ----------- 受保护接口，加上 Depends(get_current_active_user) -------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),  # 认证用户必须通过
):
    try:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        prompt = user_messages[-1].content
        
        output, error, return_code = execute_gemini_command(prompt, request.model, request.project_id)
        
        if return_code != 0:
            raise HTTPException(status_code=500, detail=f"Gemini CLI error: {error}")
        
        response_payload = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": f"gemini-cli-proxy",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ]
        }
        
        return response_payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=SimpleChatResponse)
async def simple_chat(
    request: SimpleChatRequest,
    current_user: User = Depends(get_current_active_user),
):
    try:
        output, error, return_code = execute_gemini_command(request.message, request.model, request.project_id)
        
        if return_code == 0:
            return SimpleChatResponse(
                response=output,
                status="success"
            )
        else:
            return SimpleChatResponse(
                response="",
                status="error",
                error=f"Gemini CLI 错误: {error}"
            )
            
    except Exception as e:
        return SimpleChatResponse(
            response="",
            status="error",
            error=f"服务器错误: {str(e)}"
        )

# 添加CORS支持
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
