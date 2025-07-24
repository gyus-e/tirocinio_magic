import torch
from controller.initialize_cache import initialize_cache
import controller
from utils import LLM
from test_utils import configuration, questions


def test():
    torch.set_grad_enabled(False)
    llm = LLM(configuration.model_name, configuration.tokenizer_name)
    cache_path = initialize_cache(configuration, llm)

    print("\n\tCAG\n")
    for i, question in enumerate(questions):
        print(f"Question {i}: {question}")

        cache = torch.load(cache_path, map_location=llm.device(), weights_only=False)
        cag_answer = controller.get_answer(
            question, llm.tokenizer(), llm.model(), llm.device(), cache
        )
        controller.clean_up_cache(cache)
        print(f"CAG:\n{cag_answer}\n")


if __name__ == "__main__":
    test()
