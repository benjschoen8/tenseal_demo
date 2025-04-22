import tenseal as ts
from .CVec import CVec
from .CMtx import CMtx

class Encryptor:

    def __init__(self, context=None, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 60]):
        # Create the encryption context for CKKS
        if context is None:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS, 
                poly_modulus_degree=poly_modulus_degree, 
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            
            # Set noise scale and generate keys
            self.context.global_scale = 2**26
            self.context.generate_galois_keys()

        else:
            self.context = context

    def encrypt(self, data):
        if not hasattr(self, 'context'):
            raise Exception("Encryptor not initialized.")
        
        ciphertext = ts.ckks_tensor(self.context, data)
        return ciphertext

    def is_public(self):
        return self.context.is_public()

    def get_sk(self):
        if not self.is_public():
            return self.context.secret_key()
        else:
            print("context is public, no secret_key")

    def make_public(self):
        if not self.is_public():
            self.context.make_context_public()

    def get_context(self):
        """
        Return the current encryption context.
        """
        return self.context


