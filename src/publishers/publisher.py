from abc import ABC, abstractmethod

class Publisher(ABC):
   
   @abstractmethod
   def write(self):
        pass
