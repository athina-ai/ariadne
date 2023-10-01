from abc import ABC, abstractmethod
from ...loaders.summarization_loader import SummarizationLoader

class SummarizationEvaluator(ABC):
    
    @abstractmethod
    def __init__(self, loader:SummarizationLoader, **kwargs):
        pass

    @abstractmethod
    def run(self):
        pass

