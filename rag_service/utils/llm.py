from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    def __init__(self, model_name: str):
        print("Starting LLM initialization...")

        self._device = Accelerator().device
        print(f"Accelerator device: {self._device}")

        print(f"Loading model: {model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
        ).to(self._device)
        print("Model loaded successfully.")

        print(f"Loading tokenizer: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully.")

        self._model.eval()
        print("LLM initialization complete.")

    def model(self) -> AutoModelForCausalLM:
        return self._model

    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def device(self):
        return self._device
