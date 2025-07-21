from utils import DB


__all__ = ["RagConfiguration"]


class RagConfiguration(DB.Model):
    config_id: int = DB.Column(DB.Integer, primary_key=True, autoincrement=True)
    system_prompt: str = DB.Column(DB.String, nullable=False)
    model_name: str = DB.Column(DB.String, nullable=False)
    embed_model_name: str = DB.Column(DB.String, nullable=False)
    chunk_size: int = DB.Column(DB.Integer, default=512)
    chunk_overlap: int = DB.Column(DB.Integer, default=50)
    temperature: float = DB.Column(DB.Float, default=0.2)
    top_k: int = DB.Column(DB.Integer, nullable=True)
    top_p: float = DB.Column(DB.Float, nullable=True)

    def __init__(self, 
                 system_prompt: str, 
                 model_name: str, 
                 embed_model_name: str, 
                 chunk_size: int, 
                 chunk_overlap: int, 
                 temperature: float, 
                 top_k: int, 
                 top_p: float
                ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p 