"""全局配置：pydantic-settings 从 .env / 环境变量读取"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "commerceflow"

    # Milvus
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "kb_policy"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # 检索相关
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"
    retrieval_top_k: int = 20        # 粗排召回数量
    rerank_top_n: int = 5            # 精排后保留数量
    token_budget: int = 3000         # 单次会话上下文token预算

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
