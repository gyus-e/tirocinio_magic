import torch
from controller.initialize_cache import initialize_cache
import controller
from utils import LLM
from .configuration_mock import configuration
from .questions_mock import questions

torch.set_grad_enabled(False)
documents = None
llm = LLM(configuration.model_name)
cache_path = initialize_cache(configuration, llm)

def test():
    print("\n\tCAG\n")
    for i, question in enumerate(questions):
        print(f"Question {i}: {question}")

        cache = torch.load(cache_path, weights_only=False)
        cag_answer = controller.get_answer(
            question, llm.tokenizer(), llm.model(), llm.device(), cache
        )
        controller.clean_up_cache(cache)
        print(f"CAG: {cag_answer}\n")


if __name__ == "__main__":
    test()
