#!/usr/bin/env python3
"""
Gemini CLI API 包装服务器
集成 OAuth2 密码模式 + JWT 鉴权，Token 永不过期
新增：支持多轮会话的接口 /v1/chat/sessions/{session_id}/completions
支持会话轮数限制（最多20轮），会话过期清理（10分钟），会话数量限制（最多5个）
新增：支持图片和文件上传功能
"""

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form, Request, Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import subprocess
import uuid
import datetime
import logging
import os
import tempfile
import shutil
import base64
import re
from typing import Optional, List, Dict, Union, Any, Set
from contextlib import asynccontextmanager
import uvicorn

from jose import JWTError, jwt
from passlib.context import CryptContext

# 会话存储结构
class SessionData:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.last_update: datetime.datetime = datetime.datetime.utcnow()
        self.uploaded_files: Dict[str, str] = {}  # 原文件名 -> 实际存储路径的映射

# 全局会话字典
sessions: Dict[str, SessionData] = {}

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

tags_metadata = [
    {
        "name": "系统信息",
        "description": "系统信息及健康检查",
    },
    {
        "name": "用户认证",
        "description": "获取token",
    },
    {
        "name": "对话",
        "description": "与Gemini 进行对话",
    },
    {
        "name": "会话管理",
        "description": "管理用户会话",
    },
    {
        "name": "反爬虫机制",
        "description": "反爬虫策略",
    }
]

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
        username = payload.get("sub")
        if username is None or not isinstance(username, str):
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
    # 编程语言代码文件
    "text/x-python", "application/javascript", "application/typescript", "text/x-java-source",
    "text/x-csrc", "text/x-c++src", "text/x-go", "application/x-sh", "application/x-httpd-php",
    "application/x-ruby", "text/rust",  # Rust 没有统一标准 MIME，可视实际使用情况调整

    # 其他文档格式
    "application/json", "application/pdf", "application/rtf",
    "text/plain", "text/markdown", "text/csv", "text/html", "text/xml", "application/octet-stream",

    # 办公文档格式
    "application/msword", "application/vnd.ms-excel", "application/vnd.ms-powerpoint",  # doc, xls, ppt
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

async def save_session_file(file: UploadFile, session_id: str) -> tuple[str, str]:
    """为会话保存文件，支持同名文件智能处理"""
    # 使用安全的目录，避免让gemini访问敏感的/root目录
    safe_dir = "/opt/files"  # 使用专门的安全目录
    session_temp_dir = os.path.join(safe_dir, f"session_{session_id}")
    os.makedirs(session_temp_dir, exist_ok=True)
    
    original_name = file.filename or "uploaded_file"
    
    # 检查会话中是否已存在同名文件
    if session_id in sessions and original_name in sessions[session_id].uploaded_files:
        existing_path = sessions[session_id].uploaded_files[original_name]
        
        # 检查现有文件是否还存在
        if os.path.exists(existing_path):
            # 比较文件内容是否相同
            content = await file.read()
            await file.seek(0)  # 重置文件指针
            
            try:
                with open(existing_path, 'rb') as existing_file:
                    existing_content = existing_file.read()
                
                if content == existing_content:
                    logger.info(f"文件内容相同，复用现有文件: {existing_path}")
                    return existing_path, "reused"
                else:
                    logger.info(f"同名文件内容不同，将覆盖: {original_name}")
                    # 删除旧文件
                    cleanup_temp_file(existing_path)
            except Exception as e:
                logger.warning(f"读取现有文件失败: {e}")
    
    # 生成新的文件路径
    name, ext = os.path.splitext(original_name)
    temp_path = os.path.join(session_temp_dir, original_name)
    
    try:
        # 写入文件内容
        content = await file.read()
        
        # 检查实际文件大小
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"文件大小超过限制 {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(content)
        
        # 更新会话文件映射
        if session_id in sessions:
            sessions[session_id].uploaded_files[original_name] = temp_path
        
        logger.info(f"文件已保存: {temp_path} (原名: {original_name})")
        return temp_path, "new"
    except Exception as e:
        # 清理失败的文件
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        raise e

async def save_temp_file(file: UploadFile, session_id: str = None) -> str:
    """保存临时文件并返回路径，支持会话隔离"""
    if session_id:
        # 对于会话文件，使用智能处理
        file_path, _ = await save_session_file(file, session_id)
        return file_path
    
    # 非会话文件，保存到安全目录
    safe_dir = "/opt/files"  # 使用专门的安全目录
    temp_dir = os.path.join(safe_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 生成唯一文件名，避免同名文件冲突
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_name = file.filename or "uploaded_file"
    name, ext = os.path.splitext(original_name)
    unique_filename = f"{name}_{timestamp}{ext}"
    temp_path = os.path.join(temp_dir, unique_filename)
    
    try:
        # 写入文件内容
        content = await file.read()
        
        # 检查实际文件大小
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"文件大小超过限制 {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(content)
        
        logger.info(f"文件已保存: {temp_path} (原名: {original_name})")
        return temp_path
    except Exception as e:
        # 清理失败的文件
        try:
            if os.path.exists(temp_path):
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

# ----------- 反爬虫配置 -------------------

# 已知的爬虫User-Agent模式
CRAWLER_USER_AGENTS = [
    r".*bot.*", r".*crawler.*", r".*spider.*", r".*scraper.*",
    r".*googlebot.*", r".*bingbot.*", r".*baiduspider.*", r".*yandexbot.*",
    r".*facebookexternalhit.*", r".*twitterbot.*", r".*linkedinbot.*",
    r".*whatsapp.*", r".*telegram.*", r".*slack.*", r".*discord.*",
    r".*curl.*", r".*wget.*", r".*python.*", r".*requests.*",
    r".*postman.*", r".*insomnia.*", r".*httpie.*",
    r".*java.*", r".*apache.*", r".*nginx.*", r".*php.*",
    r".*node.*", r".*go-http.*", r".*ruby.*", r".*perl.*"
]

# 可疑路径模式 
SUSPICIOUS_PATHS = [
    r"/robots\.txt", r"/sitemap\.xml", r"/favicon\.ico",
    r"/\.well-known/.*", r"/wp-admin/.*", r"/admin/.*", r"/login.*",
    r"/\.git.*", r"/\.svn.*", r"/\.env.*", r"/config.*",
    r"/backup.*", r"/test.*", r"/debug.*", r"/api/v\d+/.*"
]

# 允许的路径（白名单）
ALLOWED_PATHS = [
    r"/", r"/docs.*", r"/redoc.*", r"/openapi\.json",
    r"/health", r"/token", r"/v1/chat/.*", r"/chat"
]

class AntiCrawlerMiddleware(BaseHTTPMiddleware):
    """反爬虫中间件"""
    
    def __init__(self, app, block_mode: str = "block"):
        """
        初始化反爬虫中间件
        block_mode: 'block' - 直接拒绝, 'log' - 只记录日志但允许访问, 'rate_limit' - 限流
        """
        super().__init__(app)
        self.block_mode = block_mode
        self.blocked_ips = set()  # 被阻止的IP
        self.request_counts = {}  # IP请求计数
        self.last_reset = datetime.datetime.now()
        
    async def dispatch(self, request: Request, call_next):
        client_ip = self.get_client_ip(request)
        user_agent = request.headers.get("user-agent", "").lower()
        path = request.url.path
        
        # 检查是否是被阻止的IP
        if client_ip in self.blocked_ips:
            logger.warning(f"🚫 被阻止的IP尝试访问: {client_ip} -> {path}")
            return PlainTextResponse("Access denied", status_code=403)
        
        # 检查是否是爬虫
        is_crawler = self.is_crawler_request(user_agent, path)
        
        if is_crawler:
            logger.warning(f"🕷️ 检测到爬虫访问: {client_ip} | {user_agent[:50]}... | {path}")
            
            if self.block_mode == "block":
                # 记录可疑IP
                self.blocked_ips.add(client_ip)
                return PlainTextResponse("Access denied - Automated requests not allowed", status_code=403)
            elif self.block_mode == "rate_limit":
                # 限流处理
                if self.should_rate_limit(client_ip):
                    return PlainTextResponse("Too many requests", status_code=429)
        
        # 正常请求处理
        response = await call_next(request)
        
        # 记录可疑请求但不阻止
        if is_crawler and self.block_mode == "log":
            logger.info(f"📊 爬虫请求已记录但允许: {client_ip} -> {path}")
        
        return response
    
    def get_client_ip(self, request: Request) -> str:
        """获取客户端真实IP"""
        # 优先从代理头获取真实IP
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def is_crawler_request(self, user_agent: str, path: str) -> bool:
        """判断是否是爬虫请求"""
        # 检查User-Agent
        for pattern in CRAWLER_USER_AGENTS:
            if re.search(pattern, user_agent, re.IGNORECASE):
                return True
        
        # 检查请求路径
        for pattern in SUSPICIOUS_PATHS:
            if re.search(pattern, path, re.IGNORECASE):
                return True
        
        # 检查是否在白名单中
        for pattern in ALLOWED_PATHS:
            if re.search(pattern, path, re.IGNORECASE):
                return False
        
        return False
    
    def should_rate_limit(self, client_ip: str) -> bool:
        """检查是否应该限流"""
        now = datetime.datetime.now()
        
        # 每小时重置计数
        if (now - self.last_reset).total_seconds() > 3600:
            self.request_counts.clear()
            self.last_reset = now
        
        # 记录请求次数
        self.request_counts[client_ip] = self.request_counts.get(client_ip, 0) + 1
        
        # 超过限制则进行限流 (每小时最多10次请求)
        return self.request_counts[client_ip] > 10

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
    description="包装Gemini CLI的简单API服务，集成OAuth2密码模式 + JWT鉴权，支持图片文件上传，内置反爬虫保护",
    tags_metadata=tags_metadata,
    lifespan=lifespan
)

# ----------- 反爬虫设置 -------------------

# 反爬虫模式配置：
# "block" - 直接拒绝爬虫请求 (推荐)
# "log" - 只记录日志但允许访问 (调试用)
# "rate_limit" - 对爬虫进行限流
ANTI_CRAWLER_MODE = os.environ.get("ANTI_CRAWLER_MODE", "block")

# 添加反爬虫中间件
app.add_middleware(AntiCrawlerMiddleware, block_mode=ANTI_CRAWLER_MODE)

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
    message: str = Field(..., description="用户消息内容")
    model: Optional[str] = Field("gemini-2.5-pro", description="使用的AI模型")
    project_id: Optional[str] = Field("", description="Google Cloud项目ID，留空使用默认项目")

class SimpleChatResponse(BaseModel):
    response: str
    status: str
    error: Optional[str] = None

@app.get("/", tags=["系统信息"])
async def root():
    return {"message": "Gemini CLI API 服务器运行中", "docs": "/docs"}


@app.get("/health", tags=["系统信息"])
async def health_check():
    try:
        result = subprocess.run(["gemini", "--help"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return {"status": "healthy", "gemini_cli": "available"}
        return {"status": "unhealthy", "gemini_cli": "not available"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/token", tags=["用户认证"], response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


def execute_gemini_command(prompt: str, model: str = "gemini-2.5-pro", project_id: Optional[str] = None, file_paths: Optional[List[str]] = None) -> tuple[str, str, int]:
    """执行Gemini CLI命令，支持多个文件输入"""
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
        
        # 默认包含的目录
        include_dirs = set(["/opt/user_data", "/opt/files"])
        enhanced_prompt = prompt
        
        if file_paths:
            # 添加所有文件所在目录
            for file_path in file_paths:
                file_dir = os.path.dirname(file_path)
                include_dirs.add(file_dir)
            
            # 添加所有文件路径到提示词
            file_args = " ".join([f'"{fp}"' for fp in file_paths])
            enhanced_prompt = f"{prompt} {file_args}"
        
        # 构建包含目录参数
        include_dirs_str = " ".join([f'--include-directories "{d}"' for d in include_dirs])
        shell_command = f'gemini -m "{model}" -p "{enhanced_prompt}" {include_dirs_str}'
        
        logger.info(f"执行命令: {shell_command[:200]}...")
        
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

# ----------- 对话接口 -------------------
# 简单对话
@app.post("/chat", tags=["对话"], response_model=SimpleChatResponse)
async def simple_chat(
    message: str = Form(..., description="用户消息内容"),
    model: str = Form("gemini-2.5-pro", description="使用的AI模型"),
    project_id: Optional[str] = Form("", description="Google Cloud项目ID，留空使用默认项目"),
    files: List[UploadFile] = File([], description="可选：上传多个20MB以内的图片或文档文件（可多选）"),
    current_user: User = Depends(get_current_active_user)
):
    """简单的聊天接口，支持多文件上传"""
    temp_file_paths = []
    
    try:
        # 处理文件 - 检查文件是否真正存在且有内容
        if files:
            for file in files:
                if hasattr(file, 'filename') and file.filename and file.filename.strip():
                    try:
                        file_type = validate_file(file)
                        temp_file_path = await save_temp_file(file)
                        temp_file_paths.append(temp_file_path)
                        logger.info(f"已保存临时文件: {temp_file_path}, 类型: {file_type}")
                    except Exception as e:
                        logger.error(f"文件处理失败: {e}")
                        # 清理已保存的文件
                        for path in temp_file_paths:
                            cleanup_temp_file(path)
                        return SimpleChatResponse(
                            response="", 
                            status="error", 
                            error=f"文件处理失败: {str(e)}"
                        )
        
        # 如果有文件，添加文件描述到message
        if temp_file_paths:
            message = f"请分析这些文件。用户的问题是：{message}" if message else "请分析这些文件"
        
        # 执行Gemini命令
        output, error, return_code = execute_gemini_command(message, model, project_id, temp_file_paths if temp_file_paths else None)
        
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
        for temp_file_path in temp_file_paths:
            cleanup_temp_file(temp_file_path)

# 兼容OpenAI对话接口
@app.post("/v1/chat/completions", tags=["对话"])
async def chat_completions(
    messages: str = Form(..., description='[{"role":"user","content":"你好，请介绍一下自己"}]'),
    model: str = Form("gemini-2.5-pro", description="选择gemini模型"),
    temperature: float = Form(0.7, description="控制回复的随机性，0.0-1.0之间", ge=0.0, le=1.0),
    max_tokens: int = Form(1000, description="最大生成token数量", ge=1, le=8192),
    project_id: Optional[str] = Form("", description="Google Cloud项目ID，留空使用默认项目"),
    files: List[UploadFile] = File([], description="可选：上传20MB以内的图片或文档文件（可多选）"),
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
        
        # 处理文件 - 检查文件是否真正存在且有内容
        temp_file_paths = []
        if files:
            for file in files:
                if hasattr(file, 'filename') and file.filename and file.filename.strip():
                    try:
                        file_type = validate_file(file)
                        temp_file_path = await save_temp_file(file)
                        logger.info(f"已保存临时文件: {temp_file_path}, 类型: {file_type}")
                        temp_file_paths.append(temp_file_path)
                    except Exception as e:
                        logger.error(f"文件处理失败: {e}")
                        # 清理已保存的文件
                        for path in temp_file_paths:
                            cleanup_temp_file(path)
                        raise HTTPException(status_code=400, detail=f"文件处理失败: {str(e)}")
        
        try:
            # 如果有文件，添加文件描述到prompt
            if temp_file_paths:
                prompt = f"请分析这些文件。用户的问题是：{prompt}" if prompt else "请分析这些文件"
            
            # 执行Gemini命令
            output, error, return_code = execute_gemini_command(prompt, model, project_id, temp_file_paths if temp_file_paths else None)
            
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
            "files_processed": [f.filename for f in files] if files else None
            }
        finally:
            # 清理临时文件
            for temp_file_path in temp_file_paths:
                cleanup_temp_file(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")



# ----------- 多轮对话会话接口 -------------------

# Sessions dictionary is already defined at the top

MAX_SESSION_MESSAGES = 20      # 最多20轮对话
SESSION_TIMEOUT_SECONDS = 600  # 10分钟未更新即过期
MAX_ACTIVE_SESSIONS = 5        # 最大5个会话

def cleanup_expired_sessions():
    now = datetime.datetime.utcnow()
    expired_sessions = [sid for sid, data in sessions.items()
                        if (now - data.last_update).total_seconds() > SESSION_TIMEOUT_SECONDS]
    for sid in expired_sessions:
        logger.info(f"清理过期会话: {sid}")
        del sessions[sid]

def ensure_sessions_limit():
    if len(sessions) <= MAX_ACTIVE_SESSIONS:
        return
    # 按最后更新时间排序，删除最早的会话
    sorted_sessions = sorted(sessions.items(), key=lambda x: x[1].last_update)
    for sid, _ in sorted_sessions[:len(sessions) - MAX_ACTIVE_SESSIONS]:
        logger.info(f"清理超出数量限制会话: {sid}")
        del sessions[sid]

@app.post("/v1/chat/sessions/{session_id}/completions", tags=["对话"])
async def chat_session_completions(
    session_id: str = Path(..., description="会话ID，用于标识多轮对话"),
    messages: str = Form(..., description='[{"role":"user","content":"继续我们之前的对话"}]'),
    model: str = Form("gemini-2.5-pro", description="使用的AI模型"),
    temperature: float = Form(0.7, description="控制回复的随机性，0.0-1.0之间", ge=0.0, le=1.0),
    max_tokens: int = Form(1000, description="最大生成token数量", ge=1, le=8192),
    project_id: Optional[str] = Form("", description="Google Cloud项目ID，留空使用默认项目"),
    files: List[UploadFile] = File([], description="可选：上传多个20MB以内的图片或文档文件（可多选）"),
    current_user: User = Depends(get_current_active_user),
):
    """支持多轮会话和多文件上传的对话接口"""
    cleanup_expired_sessions()
    ensure_sessions_limit()
    
    temp_file_paths = []
    
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
            sessions[session_id] = SessionData()

        # 添加新消息到会话
        session_messages = []
        for msg in messages_list:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                session_messages.append({"role": msg["role"], "content": msg["content"]})
        
        sessions[session_id].messages.extend(session_messages)
        
        # 保持会话轮数限制
        if len(sessions[session_id].messages) > MAX_SESSION_MESSAGES:
            sessions[session_id].messages = sessions[session_id].messages[-MAX_SESSION_MESSAGES:]

        sessions[session_id].last_update = datetime.datetime.utcnow()

        # 处理文件
        current_prompt = ""
        if session_messages:
            user_messages = [msg for msg in session_messages if msg["role"] == "user"]
            if user_messages:
                current_prompt = user_messages[-1]["content"]
        
        file_statuses = {}
        if files:
            for file in files:
                if hasattr(file, 'filename') and file.filename and file.filename.strip():
                    try:
                        file_type = validate_file(file)
                        temp_file_path, file_status = await save_session_file(file, session_id)
                        temp_file_paths.append(temp_file_path)
                        
                        status_msg = {
                            "new": "已保存新文件",
                            "reused": "复用现有同名文件（内容相同）"
                        }.get(file_status, "已处理文件")
                        
                        logger.info(f"{status_msg}: {temp_file_path}, 类型: {file_type}")
                        file_statuses[file.filename] = file_status
                    except Exception as e:
                        logger.error(f"文件处理失败: {e}")
                        raise HTTPException(status_code=400, detail=f"文件处理失败: {str(e)}")

        # 如果有文件，更新提示词
        if temp_file_paths:
            current_prompt = f"请分析这些文件。用户的问题是：{current_prompt}" if current_prompt else "请分析这些文件"

        # 构造完整的对话上下文
        context_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sessions[session_id].messages])
        
        # 如果有文件，使用当前处理后的prompt；否则使用完整上下文
        final_prompt = current_prompt if temp_file_paths else context_prompt

        # 执行Gemini命令
        output, error, return_code = execute_gemini_command(final_prompt, model, project_id, temp_file_paths if temp_file_paths else None)
        
        if return_code != 0:
            raise HTTPException(status_code=500, detail=f"Gemini CLI error: {error}")

        # 将AI回复添加到会话
        sessions[session_id].messages.append({"role": "assistant", "content": output})
        sessions[session_id].last_update = datetime.datetime.utcnow()

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
                "message_count": len(sessions[session_id].messages),
                "max_messages": MAX_SESSION_MESSAGES
            },
            "files_processed": [f.filename for f in files] if files else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        # 清理临时文件
        for temp_file_path in temp_file_paths:
            cleanup_temp_file(temp_file_path)


# ----------- 会话管理接口 -------------------

@app.get("/v1/chat/sessions", tags=["会话管理"])
async def list_sessions(current_user: User = Depends(get_current_active_user)):
    """列出所有活跃会话"""
    cleanup_expired_sessions()
    
    session_info = []
    for sid, data in sessions.items():
        session_info.append({
            "session_id": sid,
            "message_count": len(data.messages),
            "last_update": data.last_update.isoformat(),
            "expires_in_seconds": max(0, SESSION_TIMEOUT_SECONDS - int((datetime.datetime.utcnow() - data.last_update).total_seconds()))
        })
    
    return {
        "sessions": session_info,
        "total_sessions": len(sessions),
        "max_sessions": MAX_ACTIVE_SESSIONS
    }

@app.get("/v1/chat/sessions/{session_id}", tags=["会话管理"])
async def get_session(session_id: str, current_user: User = Depends(get_current_active_user)):
    """获取指定会话的详细信息"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session_data = sessions[session_id]
    return {
        "session_id": session_id,
        "messages": session_data.messages,
        "message_count": len(session_data.messages),
        "last_update": session_data.last_update.isoformat(),
        "expires_in_seconds": max(0, SESSION_TIMEOUT_SECONDS - int((datetime.datetime.utcnow() - session_data.last_update).total_seconds())),
        "uploaded_files": list(session_data.uploaded_files.keys())
    }

@app.get("/v1/chat/sessions/{session_id}/files", tags=["会话管理"])
async def list_session_files(session_id: str, current_user: User = Depends(get_current_active_user)):
    """列出会话中上传的文件"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session_data = sessions[session_id]
    file_info = []
    
    for filename, filepath in session_data.uploaded_files.items():
        file_exists = os.path.exists(filepath)
        file_size = 0
        if file_exists:
            try:
                file_size = os.path.getsize(filepath)
            except:
                pass
        
        file_info.append({
            "filename": filename,
            "path": filepath,
            "exists": file_exists,
            "size_bytes": file_size,
            "size_mb": round(file_size / 1024 / 1024, 2) if file_size > 0 else 0
        })
    
    return {
        "session_id": session_id,
        "files": file_info,
        "total_files": len(file_info)
    }

@app.delete("/v1/chat/sessions/{session_id}", tags=["会话管理"])
async def delete_session(session_id: str, current_user: User = Depends(get_current_active_user)):
    """删除指定会话"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    del sessions[session_id]
    return {"message": f"会话 {session_id} 已删除"}


# ----------- 反爬虫策略 -------------------
@app.get("/robots.txt", tags=["反爬虫策略"])
async def robots_txt():
    """返回robots.txt内容，明确拒绝所有爬虫"""
    robots_content = """User-agent: *
Disallow: /

# This is a private API service
# Automated crawling, scraping, or indexing is strictly prohibited
# Violation may result in IP blocking
"""
    return PlainTextResponse(robots_content, media_type="text/plain")

@app.get("/favicon.ico", tags=["反爬虫策略"])
async def favicon():
    """返回404避免favicon请求日志"""
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/.well-known/security.txt", tags=["反爬虫策略"])
async def security_txt():
    """安全政策文件"""
    security_content = """Contact: admin@yourdomain.com
Policy: This is a private API service
Preferred-Languages: en, zh
"""
    return PlainTextResponse(security_content, media_type="text/plain")


@app.get("/admin/blocked-ips", tags=["反爬虫策略"])
async def get_blocked_ips(current_user: User = Depends(get_current_active_user)):
    """获取被阻止的IP列表"""
    # 简化的实现，直接返回默认值
    # 在实际部署中，可以通过其他方式获取中间件状态
    return {"blocked_ips": [], "total_blocked": 0, "mode": ANTI_CRAWLER_MODE}

@app.post("/admin/unblock-ip/{ip}", tags=["反爬虫策略"])
async def unblock_ip(ip: str, current_user: User = Depends(get_current_active_user)):
    """解除IP封锁"""
    # 简化的实现，返回成功消息
    # 在实际部署中，可以通过其他方式管理IP封锁
    logger.info(f"✅ 管理员请求解封IP: {ip}")
    return {"message": f"IP {ip} 解封请求已记录"}


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
    print()
    print("💡 说明：如果看到 robots.txt 404 错误，这是正常现象（搜索引擎爬虫访问）")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
