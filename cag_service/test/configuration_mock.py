from models import CagConfiguration
# from utils import DB

model_name = "meta-llama/Llama-3.2-1B-Instruct"

cag_system_prompt = """
    Sei un assistente bibliotecario. Nel contesto ti sono fornite informazioni sul catalogo della Biblioteca Pontaniana di Napoli.
    Rispondi alle domande degli utenti con le informazioni pertinenti.
""".join("\n")

configuration = CagConfiguration(cag_system_prompt, model_name)
# DB.session.add(configuration)
# DB.session.commit()
