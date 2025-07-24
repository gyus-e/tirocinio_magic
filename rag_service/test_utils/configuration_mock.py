from models import RagConfiguration

# from utils import DB


model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

rag_system_prompt = """
    Sei un assistente bibliotecario. Hai accesso a una serie di documenti contenenti informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Cerca nei documenti per trovare le risposte alle domande degli utenti.
    Qualora non ci siano, rispondi "Non lo so".
""".join(
    "\n"
)

embed_model_name = "BAAI/bge-m3"
chunk_size = 1024
chunk_overlap = 128
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
