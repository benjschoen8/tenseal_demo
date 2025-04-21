from abc import ABC, abstractmethod

class Cipher(ABC):
    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def decrypt(self):
        pass


