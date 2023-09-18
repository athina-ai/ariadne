from abc import ABC, abstractmethod
import json


class Loader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, filename: str) -> None:
        """Load data in the specified format."""
        pass

    @abstractmethod
    def process(self) -> None:
        """Prepare dataset to be consumed by evaluators."""
        pass
