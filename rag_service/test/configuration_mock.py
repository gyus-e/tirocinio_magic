from models import RagConfiguration
# from utils import DB


model_name = "meta-llama/Llama-3.2-1B-Instruct"

embed_model_name = "BAAI/bge-base-en-v1.5"

rag_system_prompt = """
    Sei un assistente bibliotecario. Hai accesso a una serie di documenti contenenti informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Rispondi alle domande degli utenti cercando nei documenti le informazioni pertinenti.
""".join("\n")

chunk_size = 512
chunk_overlap = 64
temperature = 0.4

configuration = RagConfiguration(rag_system_prompt, model_name, embed_model_name, chunk_size, chunk_overlap, temperature)
# DB.session.add(configuration)
# DB.session.commit()
