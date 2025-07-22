from models import CagConfiguration

# from utils import DB

cache_name = "test_cache"

model_name = "microsoft/Phi-4-mini-instruct"

cag_system_prompt = """
    Sei un assistente bibliotecario. Nel contesto ti sono fornite informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Rispondi alle domande con le informazioni pertinenti.
""".join(
    "\n"
)

configuration = CagConfiguration(cag_system_prompt, model_name, cache_name)
# DB.session.add(configuration)
# DB.session.commit()
