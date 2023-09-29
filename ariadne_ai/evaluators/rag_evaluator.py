from abc import ABC, abstractmethod
from ..loaders.rag_generation_loader import RagGenerationLoader

class RagEvaluator(ABC):
    
    @abstractmethod
    def __init__(self, loader:RagGenerationLoader, **kwargs):
        pass

    @abstractmethod
    def run(self):
        pass

