from abc import ABC, abstractmethod
from ...loaders.rag_loader import RagLoader


class RagEvaluator(ABC):
    @abstractmethod
    def __init__(self, loader: RagLoader, **kwargs):
        pass

    @abstractmethod
    def run(self):
        pass
