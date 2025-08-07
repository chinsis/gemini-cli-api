#!/usr/bin/env python3
"""
Gemini CLI API 包装服务器
集成 OAuth2 密码模式 + JWT 鉴权，Token 永不过期
新增：支持多轮会话的接口 /v1/chat/sessions/{session_id}/completions
支持会话轮数限制（最多20轮），会话过期清理（10分钟），会话数量限制（最多5个）
新增：支持图片和文件上传功能
"""

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import subprocess
import uuid
import datetime
import logging
import os
import tempfile
import shutil
import base64
from typing import Optional, List, Dict, Union
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
    "mosh": {
        "username": "mosh",
        "full_name": "Xuu",
        "email": "mosh@example.com",
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

# ----------- 文件处理工具 -------------------

# 支持的图片格式
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", 
    "image/bmp", "image/tiff", "image/svg+xml"
}

# 支持的文档格式
SUPPORTED_DOCUMENT_TYPES = {
    "text/plain", "text/markdown", "text/csv", "text/html", "text/xml",
    "application/json", "application/pdf", "application/rtf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"  # pptx
}

# 最大文件大小 (20MB)
MAX_FILE_SIZE = 20 * 1024 * 1024

def validate_file(file: UploadFile) -> str:
    """验证上传的文件"""
    if not file.content_type:
        raise HTTPException(status_code=400, detail="无法确定文件类型")
    
    if file.content_type not in SUPPORTED_IMAGE_TYPES and file.content_type not in SUPPORTED_DOCUMENT_TYPES:
        supported_types = list(SUPPORTED_IMAGE_TYPES) + list(SUPPORTED_DOCUMENT_TYPES)
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件类型: {file.content_type}. 支持的类型: {', '.join(supported_types)}"
        )
    
    # 检查文件大小
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"文件大小超过限制 {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    return "image" if file.content_type in SUPPORTED_IMAGE_TYPES else "document"

async def save_temp_file(file: UploadFile) -> str:
    """保存临时文件并返回路径"""
    # 创建临时文件
    suffix = ""
    if file.filename:
        suffix = os.path.splitext(file.filename)[1]
    
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    
    try:
        # 写入文件内容
        content = await file.read()
        
        # 检查实际文件大小
        if len(content) > MAX_FILE_SIZE:
            os.close(temp_fd)
            os.unlink(temp_path)
            raise HTTPException(
                status_code=413, 
                detail=f"文件大小超过限制 {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        with os.fdopen(temp_fd, 'wb') as temp_file:
            temp_file.write(content)
        
        return temp_path
    except Exception as e:
        # 清理失败的文件
        try:
            os.close(temp_fd)
            os.unlink(temp_path)
        except:
            pass
        raise e

def cleanup_temp_file(file_path: str):
    """清理临时文件"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"已清理临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"清理临时文件失败 {file_path}: {e}")

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
    description="包装Gemini CLI的简单API服务，集成OAuth2密码模式 + JWT鉴权，支持图片文件上传",
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

def execute_gemini_command(prompt: str, model: str = "gemini-2.5-pro", project_id: str = None, file_path: str = None) -> tuple[str, str, int]:
    """执行Gemini CLI命令，支持文件输入"""
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
        
        # 构建命令
        if file_path:
            # 有文件时，使用文件作为输入
            shell_command = f'gemini -m "{model}" -p "{prompt}" < "{file_path}"'
        else:
            # 没有文件时，使用原来的方式
            shell_command = f'echo "" | gemini -m "{model}" -p "{prompt}"'
        
        logger.info(f"执行命令: {shell_command[:100]}...")
        
        result = subprocess.run(
            shell_command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=120,  # 文件处理可能需要更长时间
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

# ----------- 支持文件上传的对话接口 -------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    messages: str = Form(..., description="JSON格式的消息数组"),
    model: str = Form("gemini-2.5-pro"),
    temperature: float = Form(0.7),
    max_tokens: int = Form(1000),
    project_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None, description="可选：上传的图片或文档文件"),
    current_user: User = Depends(get_current_active_user)
):
    """OpenAI兼容的聊天完成接口，支持文件上传"""
    try:
        # 解析消息
        import json
        try:
            messages_list = json.loads(messages)
            if not isinstance(messages_list, list):
                raise ValueError("messages必须是数组格式")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="messages格式错误，必须是有效的JSON数组")
        
        # 获取用户消息
        user_messages = [msg for msg in messages_list if msg.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        prompt = user_messages[-1].get("content", "")
        
        # 处理文件
        temp_file_path = None
        if file:
            file_type = validate_file(file)
            temp_file_path = await save_temp_file(file)
            logger.info(f"已保存临时文件: {temp_file_path}, 类型: {file_type}")
            
            # 为文件添加描述到prompt
            if file_type == "image":
                prompt = f"请分析这张图片。用户的问题是：{prompt}" if prompt else "请描述这张图片的内容"
            else:
                prompt = f"请分析这个文档。用户的问题是：{prompt}" if prompt else "请总结这个文档的内容"
        
        try:
            # 执行Gemini命令
            output, error, return_code = execute_gemini_command(prompt, model, project_id, temp_file_path)
            
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
                }],
                "file_processed": file.filename if file else None
            }
        finally:
            # 清理临时文件
            if temp_file_path:
                cleanup_temp_file(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/chat", response_model=SimpleChatResponse)
async def simple_chat(
    message: str = Form(..., description="用户消息"),
    model: str = Form("gemini-2.5-pro"),
    project_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None, description="可选：上传的图片或文档文件"),
    current_user: User = Depends(get_current_active_user)
):
    """简单的聊天接口，支持文件上传"""
    temp_file_path = None
    
    try:
        # 处理文件
        if file:
            file_type = validate_file(file)
            temp_file_path = await save_temp_file(file)
            logger.info(f"已保存临时文件: {temp_file_path}, 类型: {file_type}")
            
            # 为文件添加描述到message
            if file_type == "image":
                message = f"请分析这张图片。用户的问题是：{message}" if message else "请描述这张图片的内容"
            else:
                message = f"请分析这个文档。用户的问题是：{message}" if message else "请总结这个文档的内容"
        
        # 执行Gemini命令
        output, error, return_code = execute_gemini_command(message, model, project_id, temp_file_path)
        
        if return_code == 0:
            return SimpleChatResponse(
                response=output, 
                status="success",
                error=None
            )
        else:
            return SimpleChatResponse(
                response="", 
                status="error", 
                error=f"Gemini CLI 错误: {error}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        return SimpleChatResponse(
            response="", 
            status="error", 
            error=f"服务器错误: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

# ----------- 多轮对话会话接口 -------------------

# 会话存储结构
sessions: Dict[str, Dict[str, object]] = {}

MAX_SESSION_MESSAGES = 20      # 最多20轮对话
SESSION_TIMEOUT_SECONDS = 600  # 10分钟未更新即过期
MAX_ACTIVE_SESSIONS = 5        # 最大5个会话

def cleanup_expired_sessions():
    now = datetime.datetime.utcnow()
    expired_sessions = [sid for sid, data in sessions.items()
                        if (now - data["last_update"]).total_seconds() > SESSION_TIMEOUT_SECONDS]
    for sid in expired_sessions:
        logger.info(f"清理过期会话: {sid}")
        del sessions[sid]

def ensure_sessions_limit():
    if len(sessions) <= MAX_ACTIVE_SESSIONS:
        return
    # 按最后更新时间排序，删除最早的会话
    sorted_sessions = sorted(sessions.items(), key=lambda x: x[1]["last_update"])
    for sid, _ in sorted_sessions[:len(sessions) - MAX_ACTIVE_SESSIONS]:
        logger.info(f"清理超出数量限制会话: {sid}")
        del sessions[sid]

@app.post("/v1/chat/sessions/{session_id}/completions")
async def chat_session_completions(
    session_id: str,
    messages: str = Form(..., description="JSON格式的消息数组"),
    model: str = Form("gemini-2.5-pro"),
    temperature: float = Form(0.7),
    max_tokens: int = Form(1000),
    project_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None, description="可选：上传的图片或文档文件"),
    current_user: User = Depends(get_current_active_user),
):
    """支持多轮会话的对话接口，支持文件上传"""
    cleanup_expired_sessions()
    ensure_sessions_limit()
    
    temp_file_path = None
    
    try:
        # 解析消息
        import json
        try:
            messages_list = json.loads(messages)
            if not isinstance(messages_list, list):
                raise ValueError("messages必须是数组格式")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="messages格式错误，必须是有效的JSON数组")
        
        # 初始化或获取会话
        if session_id not in sessions:
            if len(sessions) >= MAX_ACTIVE_SESSIONS:
                raise HTTPException(status_code=429, detail="会话数量已达上限，请稍后重试")
            sessions[session_id] = {"messages": [], "last_update": datetime.datetime.utcnow()}

        # 添加新消息到会话
        session_messages = []
        for msg in messages_list:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                session_messages.append({"role": msg["role"], "content": msg["content"]})
        
        sessions[session_id]["messages"].extend(session_messages)
        
        # 保持会话轮数限制
        if len(sessions[session_id]["messages"]) > MAX_SESSION_MESSAGES:
            sessions[session_id]["messages"] = sessions[session_id]["messages"][-MAX_SESSION_MESSAGES:]

        sessions[session_id]["last_update"] = datetime.datetime.utcnow()

        # 处理文件
        current_prompt = ""
        if session_messages:
            user_messages = [msg for msg in session_messages if msg["role"] == "user"]
            if user_messages:
                current_prompt = user_messages[-1]["content"]
        
        if file:
            file_type = validate_file(file)
            temp_file_path = await save_temp_file(file)
            logger.info(f"已保存临时文件: {temp_file_path}, 类型: {file_type}")
            
            # 为文件添加描述
            if file_type == "image":
                current_prompt = f"请分析这张图片。用户的问题是：{current_prompt}" if current_prompt else "请描述这张图片的内容"
            else:
                current_prompt = f"请分析这个文档。用户的问题是：{current_prompt}" if current_prompt else "请总结这个文档的内容"

        # 构造完整的对话上下文
        context_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sessions[session_id]["messages"]])
        
        # 如果有文件，使用当前处理后的prompt；否则使用完整上下文
        final_prompt = current_prompt if file else context_prompt

        # 执行Gemini命令
        output, error, return_code = execute_gemini_command(final_prompt, model, project_id, temp_file_path)
        
        if return_code != 0:
            raise HTTPException(status_code=500, detail=f"Gemini CLI error: {error}")

        # 将AI回复添加到会话
        sessions[session_id]["messages"].append({"role": "assistant", "content": output})
        sessions[session_id]["last_update"] = datetime.datetime.utcnow()

        return {
            "id": str(uuid.uuid4()),
            "object": "chat.session.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": "gemini-cli-proxy",
            "session_id": session_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "session_info": {
                "message_count": len(sessions[session_id]["messages"]),
                "max_messages": MAX_SESSION_MESSAGES
            },
            "file_processed": file.filename if file else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        # 清理临时文件
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

# ----------- 会话管理接口 -------------------

@app.get("/v1/chat/sessions")
async def list_sessions(current_user: User = Depends(get_current_active_user)):
    """列出所有活跃会话"""
    cleanup_expired_sessions()
    
    session_info = []
    for sid, data in sessions.items():
        session_info.append({
            "session_id": sid,
            "message_count": len(data["messages"]),
            "last_update": data["last_update"].isoformat(),
            "expires_in_seconds": max(0, SESSION_TIMEOUT_SECONDS - int((datetime.datetime.utcnow() - data["last_update"]).total_seconds()))
        })
    
    return {
        "sessions": session_info,
        "total_sessions": len(sessions),
        "max_sessions": MAX_ACTIVE_SESSIONS
    }

@app.delete("/v1/chat/sessions/{session_id}")
async def delete_session(session_id: str, current_user: User = Depends(get_current_active_user)):
    """删除指定会话"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    del sessions[session_id]
    return {"message": f"会话 {session_id} 已删除"}

@app.get("/v1/chat/sessions/{session_id}")
async def get_session(session_id: str, current_user: User = Depends(get_current_active_user)):
    """获取指定会话的详细信息"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session_data = sessions[session_id]
    return {
        "session_id": session_id,
        "messages": session_data["messages"],
        "message_count": len(session_data["messages"]),
        "last_update": session_data["last_update"].isoformat(),
        "expires_in_seconds": max(0, SESSION_TIMEOUT_SECONDS - int((datetime.datetime.utcnow() - session_data["last_update"]).total_seconds()))
    }

# ----------- CORS 和启动 -------------------

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    print("🚀 启动 Gemini CLI API 服务器（带OAuth2/JWT认证 + 文件上传支持）...")
    print("📖 API 文档: http://localhost:8000/docs")
    print("🔗 健康检查: http://localhost:8000/health")
    print("🔑 获取Token接口: http://localhost:8000/token")
    print("📁 支持文件类型:")
    print(f"   图片: {', '.join(SUPPORTED_IMAGE_TYPES)}")
    print(f"   文档: {', '.join(SUPPORTED_DOCUMENT_TYPES)}")
    print(f"📏 最大文件大小: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)