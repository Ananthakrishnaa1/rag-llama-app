from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "test"
    
    class Config:
        env_file = ".env"

settings = Settings()