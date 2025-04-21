from .Cipher import Cipher
import numpy as np

class CVec(Cipher):
    _ciphertext = None
    _context = None

    def __init__(self, ciphertext):
        self._ciphertext = ciphertext

    def __add__(self, other):
        obj = self.__new__(CVec)
        obj._ciphertext = (self._ciphertext + other._ciphertext)
        return obj

    def __sub__(self, other):
        obj = self.__new__(CVec)
        obj._ciphertext = (self._ciphertext - other._ciphertext)
        return obj

    def __mul__(self, other):
        obj = self.__new__(CVec)
        obj._ciphertext = (self._ciphertext * other._ciphertext)
        return obj

    def data(self):
        arr = np.array([])
        for i in range(len(self._ciphertext.ciphertext())):
            arr = np.append(arr, self._ciphertext.ciphertext()[i].data())
        return arr

    def decrypt(self, secret_key = None):
        if secret_key == None:
            return self._ciphertext.decrypt().tolist()
        else:
            return self._ciphertext.decrypt(secret_key).tolist()

