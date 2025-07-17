from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "ops-iq"
    app_version: str = "0.1.0"
    debug: bool = True

    # API Settings
    master_agent_host: str = "localhost"
    master_agent_port: int = 8000
    voice_agent_host: str = "localhost"
    voice_agent_port: int = 8001
    action_agent_host: str = "localhost"
    action_agent_port: int = 8002

    # Database
    mongodb_url: str = "mongodb://localhost:27017/ops_iq"
    redis_url: str = "redis://:Infra0n@127.0.0.1:6379/0"

    # Ollama
    ollama_url: str = "http://10.0.4.211:11434"
    ollama_model: str = "mistral:7b-instruct-v0.2"

    # Audio
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 1024
    audio_channels: int = 1

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # # External APIs
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    microsoft_client_id: Optional[str] = None
    microsoft_client_secret: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
