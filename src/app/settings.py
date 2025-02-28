#using pydantic-settings

import os 
from functools import lru_cache

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = os.environ["GROQ_API_KEY"]
    openai_api_key: str = os.environ["OPENAI_API_KEY"]
    weatherstack_api: str = os.environ["WEATHER_API_KEY"]


@lru_cache

def get_settings() -> Settings:
    return Settings()



settings = get_settings()
print(settings.openai_api_key)
