from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # App
    app_name: str = "CEREP"
    app_env: str = "development"
    secret_key: str = "changeme-super-secret-key-32chars!!"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Database
    database_url: str = "sqlite:///./cerep.db"

    # LLM — Stage 1 (vLLM constrained decoding)
    vllm_base_url: str = "http://localhost:8001"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_temperature: float = 0.1
    vllm_max_tokens: int = 512
    vllm_timeout: int = 120

    # LLM — Stage 2 (Fusion decoder)
    fusion_provider: str = "openai"  # openai | ollama | vllm
    fusion_base_url: str = "https://api.openai.com/v1"
    fusion_model: str = "gpt-4.1"
    fusion_api_key: str = ""
    fusion_temperature: float = 0.3
    fusion_max_tokens: int = 1024
    fusion_timeout: int = 180

    # Legacy Ollama (backward compat)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3:8b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 256
    llm_timeout: int = 120

    # Knowledge Graph
    kg_mode: str = "networkx"       # networkx | neo4j
    kg_build_mode: str = "adapters"  # seed | adapters
    kg_cache_path: str = "backend/graph/graph_cache.json"
    kg_max_hops: int = 4

    # Neo4j (when kg_mode = "neo4j")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
