from models import RagConfiguration

# from utils import DB


model_name = "microsoft/Phi-4-mini-instruct"

embed_model_name = "BAAI/bge-m3"

rag_system_prompt = """
    Sei un assistente bibliotecario. Hai accesso a una serie di documenti contenenti informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Cerca nei documenti per trovare le risposte alle domande degli utenti.
    Qualora non ci siano, rispondi "Non lo so".
""".join(
    "\n"
)

chunk_size = 512
chunk_overlap = 64
temperature = 0.4

configuration = RagConfiguration(
    rag_system_prompt,
    model_name,
    embed_model_name,
    chunk_size,
    chunk_overlap,
    temperature,
)
# DB.session.add(configuration)
# DB.session.commit()
