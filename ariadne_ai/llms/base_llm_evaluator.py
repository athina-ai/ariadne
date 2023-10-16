from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv
import os
from .open_ai_completion import OpenAICompletion

load_dotenv()


class BaseLlmEvaluator(ABC):
    def __init__(
        self,
        model: str,
        open_ai_key: str,
        athina_api_key: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.metadata = metadata
        self.open_ai_key = (
            open_ai_key if open_ai_key is not None else os.getenv("OPENAI_API_KEY")
        )
        if self.open_ai_key is None:
            raise ValueError(
                "You must provide an OpenAI API key or set the OPENAI_API_KEY environment variable."
            )

        self.open_ai_completion = OpenAICompletion(
            model=model,
            open_ai_key=self.open_ai_key,
            athina_api_key=athina_api_key,
            metadata=metadata,
        )
