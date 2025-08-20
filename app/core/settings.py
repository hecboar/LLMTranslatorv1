# app/core/settings.py
from pydantic import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
TERMINOLOGY_BASE_DIR: str = "data/terminology"


class Config:
env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
s = Settings()
Path(s.TERMINOLOGY_BASE_DIR).mkdir(parents=True, exist_ok=True)
return s