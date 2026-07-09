from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://ddi_user:ddi_password@localhost:5432/ddi_db"
    database_sync_url: str = "postgresql://ddi_user:ddi_password@localhost:5432/ddi_db"

    # App
    app_env: str = "development"
    app_secret_key: str = "dev-secret-key"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Model paths (relative to backend/ dir)
    model_path: str = "model_artifacts/rf_model.pkl"
    model_meta_path: str = "model_artifacts/model_meta.json"

    # Data paths — relative to backend/ OR absolute on cloud
    merged_data_path: str = "../dataset/merged_data/merged_data.csv"
    train_data_path: str = "../dataset/train_data/train_set.csv"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def cors_origin_regex(self) -> str:
        return r"^https://.*\.vercel\.app$|^http://localhost:\d+$|^http://127\.0\.0\.1:\d+$"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
