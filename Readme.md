# Gemini CLI API

## 项目简介

Gemini CLI API 是一个基于 FastAPI 的包装服务器，集成了 OAuth2 密码模式和 JWT 鉴权，Token 永不过期。它支持与 Gemini CLI 进行会话，具备以下主要功能：

- **OAuth2/JWT 用户认证**：安全的 Token 登录机制。
- **多轮会话聊天接口**：支持最多 20 轮对话，自动会话过期清理（10分钟），最多支持 5 个活跃会话。
- **图片和文件上传**：支持多种图片和文档类型，单文件最大 20MB。
- **反爬虫保护**：内置反爬虫中间件，支持阻止、记录和限流爬虫请求。
- **会话管理接口**：查询、删除会话及上传文件列表。
- **OpenAI 兼容接口**：部分接口兼容 OpenAI API 格式，方便集成。
- **健康检查与系统信息接口**。

## 快速部署指南

> **推荐使用 Python 3.12 及以上版本。请先确保已安装 [Python 3.12+](https://www.python.org/downloads/)。**

1. **克隆项目代码**
   ```shell
   git clone https://github.com/chinsis/gemini-cli-api.git
   cd gemini-cli-api
   ```

2. **创建 Python 虚拟环境**
   ```shell
   python -m venv .venv
   ```

3. **激活虚拟环境**
- Windows:
    ```shell
     .venv\Scripts\activate
     ```
- macOS/Linux:
     ```shell
     source .venv/bin/activate
     ```

4. **安装依赖包**
   ```shell
   pip install -r requirements.txt
   ```

5. **配置环境变量**
    
    .env.example文件名修改为.env，具体配置参考[环境变量配置](#环境变量配置)

6. **启动服务**
   ```shell
   python main.py
   ```

7. **访问 API 文档**
- [http://your-ip:8000/docs](http://your-ip:8000/docs) （Swagger UI）

## 环境变量配置

- `GOOGLE_CLOUD_PROJECT`：默认 Google Cloud 项目 ID
- `JWT_SECRET_KEY`：JWT 加密密钥
- `PASSWORD`：默认用户密码
- `ANTI_CRAWLER_MODE`：反爬虫模式（block/log/rate_limit）

## 其他说明

- 默认用户：`mosh`，密码可通过环境变量设置。
- 文件上传目录为 `/opt/files`，请确保有写入权限。
- 详细接口说明请参考 API 文档页