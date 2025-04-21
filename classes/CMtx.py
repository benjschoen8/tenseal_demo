from .Cipher import Cipher
import numpy as np

class CMtx(Cipher):
    _ciphertext = None

    def __init__(self, ciphertext):
        self._ciphertext = ciphertext

    def __add__(self, other):
        obj = self.__new__(CMtx)
        obj._ciphertext = (self._ciphertext + other._ciphertext)
        return obj

    def __mul__(self, other):
        obj = self.__new__(CMtx)
        obj._ciphertext = (self._ciphertext * other._ciphertext)
        return obj

    def data(self):
        arr = np.array([])
        for i in range(self._ciphertext.shape[0] * self._ciphertext.shape[1]):
            arr = np.append(arr, self._ciphertext.ciphertext()[i].data())
        return arr.reshape((self._ciphertext.shape[0], self._ciphertext.shape[1]))

    def decrypt(self, secret_key):
        if secret_key == None:
            return self._ciphertext.decrypt().tolist()
        else:
            return self._ciphertext.decrypt(secret_key).tolist()

