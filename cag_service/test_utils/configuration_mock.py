from models import CagConfiguration

# from utils import DB

cache_name = "test_cache"

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.2"

cag_system_prompt = """
    Sei un assistente bibliotecario. Nel contesto ti sono fornite informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Rispondi alle domande con le informazioni pertinenti.
""".join(
    "\n"
)

configuration = CagConfiguration(cag_system_prompt, model_name, tokenizer_name, cache_name)
# DB.session.add(configuration)
# DB.session.commit()
