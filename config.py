from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_url: str = ""
    deepseek_api_key: str = ""
    tavily_api_key: str = ""
    api_key: str = ""

    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_pool_timeout: float = 10.0

    chunk_size: int = 500
    chunk_overlap: int = 80
    retrieve_k: int = 5
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "fastapi_docs_hybrid"
    milvus_search_mode: Literal["dense", "bm25", "hybrid"] = "dense"
    milvus_candidate_k: int = 10
    milvus_rrf_k: int = 60
    milvus_analyzer: Literal["standard", "english", "chinese"] = "chinese"

    reranker_enabled: bool = True
    reranker_model_path: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_candidate_k: int = 8
    reranker_device: Literal["cpu", "cuda", "mps"] = "cpu"
    reranker_timeout_seconds: float = 10.0

    max_rewrites: int = 3

    max_history_messages: int = 20

    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    retrieval_timeout_seconds: float = 20.0

    llm_model: str = "deepseek:deepseek-chat"
    grader_model: str = "deepseek:deepseek-chat"

    langsmith_tracing: bool = False
    langsmith_project: str = "agentic-rag"

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "http://localhost:3000"
    langfuse_tracing_environment: str = "development"
    cors_origins: str = "*"
    rate_limit: str = "20/minute"
    auth_fail_rate_limit: str = "10/minute"
    mcp_workspace_dir: str = "./mcp_workspace"
    mcp_enabled: bool = True
    embedding_model_path: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def require_agent_runtime(self) -> None:
        """聊天服务启动前验票；只建索引时不调用这个函数。"""
        required = {
            "POSTGRES_URL": self.postgres_url,
            "DEEPSEEK_API_KEY": self.deepseek_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            "API_KEY": self.api_key,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if self.langfuse_enabled:
            langfuse_required = {
                "LANGFUSE_PUBLIC_KEY": self.langfuse_public_key,
                "LANGFUSE_SECRET_KEY": self.langfuse_secret_key,
                "LANGFUSE_BASE_URL": self.langfuse_base_url,
            }
            missing.extend(name for name, value in langfuse_required.items() if not value.strip())
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Agent 服务缺少必填配置：{names}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
