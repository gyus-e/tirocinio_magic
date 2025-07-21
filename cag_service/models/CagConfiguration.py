from utils import DB
from sqlalchemy.dialects.postgresql import UUID
import uuid

__all__ = ["CagConfiguration"]


class CagConfiguration(DB.Model):
    config_id: int = DB.Column(DB.Integer, primary_key=True, autoincrement=True)
    system_prompt: str = DB.Column(DB.String, nullable=False)
    model_name: str = DB.Column(DB.String, nullable=False)
    cache_name: str = DB.Column(UUID(as_uuid=True), nullable=False)


    def __init__(self, system_prompt: str, model_name: str):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.cache_name = str(uuid.uuid4())