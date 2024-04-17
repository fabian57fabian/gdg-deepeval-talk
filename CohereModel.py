from typing import Optional

from cohere import Client
from deepeval.models.base_model import DeepEvalBaseLLM


class CohereModel(DeepEvalBaseLLM):
    def __init__(
            self,
            cohere_api_key: str,
            model_name: Optional[str] = "command-r",
            max_tokens: Optional[int] = 1024,
            temperature: Optional[float] = 0.7,
    ):
        self.cohere_api_key = cohere_api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__()

    def load_model(self):
        return Client(self.cohere_api_key)

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.chat(message=prompt,
                               max_tokens=self.max_tokens,
                               temperature=self.temperature).text

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = chat_model.chat(message=prompt,
                                    max_tokens=self.max_tokens,
                                    temperature=self.temperature)
        return res.text

    def get_model_name(self):
        return self.model_name
