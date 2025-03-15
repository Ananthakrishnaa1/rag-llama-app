from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "test"
    
    class Config:
        # Look for .env file in project root
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')

settings = Settings()