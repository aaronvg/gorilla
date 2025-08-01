import os
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.model_style import ModelStyle
from openai import OpenAI


class MoonshotHandler(OpenAICompletionsHandler):
    def __init__(self, model_name: str, temperature: float) -> None:
        super().__init__(model_name, temperature)

        self.client = OpenAI(
            base_url="https://api.moonshot.ai/v1",
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
