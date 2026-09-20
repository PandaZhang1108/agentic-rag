import os

os.environ.setdefault("POSTGRES_URL", "postgresql://test@localhost/test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("MCP_ENABLED", "false")
