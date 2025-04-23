import tenseal as ts
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from classes import *

def add(a, b):
    if isinstance(a, list) and isinstance(b, list):
        # Recursively add each pair of elements
        return [add(x, y) for x, y in zip(a, b)]
    else:
        return a + b

def sub(a, b):
    if isinstance(a, list) and isinstance(b, list):
        # Recursively add each pair of elements
        return [sub(x, y) for x, y in zip(a, b)]
    else:
        return a - b

def mul(a, b):
    if isinstance(a, list) and isinstance(b, list):
        # Recursively add each pair of elements
        return [mul(x, y) for x, y in zip(a, b)]
    else:
        return a * b

def matmul(A, B):
    # A is m x n, B is n x p → result is m x p
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must equal number of rows in B")

    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            # Compute dot product of A[i] row and B column j
            value = sum(A[i][k] * B[k][j] for k in range(len(B)))
            row.append(value)
        result.append(row)
    return result

def raw_data(ciphertext):
    arr = [] 
    if len(ciphertext.shape) == 2:
        for i in range(ciphertext.shape[0]):
            inn_arr = []
            for j in range(ciphertext.shape[1]):
                inn_arr.append(ciphertext.ciphertext()[i*ciphertext.shape[1] + j].data())
            arr.append(inn_arr)
    elif len(ciphertext.shape) == 1:
        for i in range(ciphertext.shape[0]):
            arr.append(ciphertext.ciphertext()[i].data())
    return arr

def demo1(input1, encryptor, sk):
    print("\033[31mSingle encrypted vector calculation\033[0m")
    vec = [3.0, 2.0, 1.0]
    print("Input vector:\n", input1)
    print("Plain vector:\n", vec)

    enc_input1 = encryptor.encrypt(input1)
    print("Input vector after encryption:\n", raw_data(enc_input1))
    print("\n")
    print("Addition\n")
    result = enc_input1 + vec
    print("calculation:\n{} + {} = {}".format(raw_data(enc_input1), vec, raw_data(result)))
    print("Decryption:\n{} -> {}".format(raw_data(result), result.decrypt(sk).tolist()))
    print("Plain calculation:", add(input1, vec))
    print("\n")
    print("Subtraction\n")
    result = enc_input1 - vec
    print("calculation:\n{} - {} = {}".format(raw_data(enc_input1), vec, raw_data(result)))
    print("Decryption:\n{} -> {}".format(raw_data(result), result.decrypt(sk).tolist()))
    print("Plain calculation:", sub(input1, vec))
    print("\n")
    print("Multiplication\n")
    result = enc_input1 * vec
    print("calculation:\n{} * {} = {}".format(raw_data(enc_input1), vec, raw_data(result)))
    print("Decryption:\n{} -> {}".format(raw_data(result), result.decrypt(sk).tolist()))
    print("Plain calculation:", mul(input1, vec))
    print("\n")

    

def demo2(input1, input2, encryptor, sk):
    print("\033[31mPure encrypted vector calculation\033[0m")
    print("1st input vector:\n", input1)
    print("2nd input vector:\n", input2)
    enc_input1 = encryptor.encrypt(input1)
    enc_input2 = encryptor.encrypt(input2)
    print("1st input vector after encryption:\n", raw_data(enc_input1))
    print("2nd input vector after encryption:\n", raw_data(enc_input2))
    print("\n")
    print("Addition\n")
    result = enc_input1 + enc_input2
    print("calculation:\n{} + {} = {}".format(raw_data(enc_input1), raw_data(enc_input2), raw_data(result)))
    print("Decryption:\n{} -> {}".format(raw_data(result), result.decrypt(sk).tolist()))
    print("Plain calculation:", add(input1, input2))
    print("\n")
    print("Subtraction\n")
    result = enc_input1 - enc_input2
    print("calculation:\n{} - {} = {}".format(raw_data(enc_input1), raw_data(enc_input2), raw_data(result)))
    print("Decryption:\n{} -> {}".format(raw_data(result), result.decrypt(sk).tolist()))
    print("Plain calculation:", sub(input1, input2))
    print("\n")
    print("Multiplication\n")
    result = enc_input1 * enc_input2
    print("calculation:\n{} * {} = {}".format(raw_data(enc_input1), raw_data(enc_input2), raw_data(result)))
    print("Decryption:\n{} -> {}".format(raw_data(result), result.decrypt(sk).tolist()))
    print("Plain calculation:", mul(input1, input2))
    print("\n")

def demo3(input1, input2, encryptor, sk):
    print("\033[31mPure encrypted matrix calculation\033[0m")
    print("1st input matrix:\n{}\n{}".format(input1[0], input1[1]))
    print("2nd input matrix:\n{}\n{}".format(input2[0], input2[1]))
    enc_input1 = encryptor.encrypt(input1)
    enc_input2 = encryptor.encrypt(input2)
    enc_mtx1 = "{}\n{}".format(raw_data(enc_input1)[0], raw_data(enc_input1)[1])
    enc_mtx2 = "{}\n{}".format(raw_data(enc_input2)[0], raw_data(enc_input2)[1])
    print("1st input matrix after encryption:\n", enc_mtx1)
    print("2nd input matrix after encryption:\n", enc_mtx2)
    print("\n")
    print("Addtion\n")
    result = enc_input1 + enc_input2
    enc_result = "{}\n{}".format(raw_data(result)[0], raw_data(result)[1])
    print("Encrypted result:\n", enc_result)
    dec_result = result.decrypt(sk).tolist()
    print("Decryption:\n{}\n{}".format(dec_result[0], dec_result[1]))
    plain_mtx = add(input1, input2)
    print("Plain calculation:\n{}\n{}".format(plain_mtx[0], plain_mtx[1]))
    print("\n")
    print("Subtraction\n")
    result = enc_input1 - enc_input2
    enc_result = "{}\n{}".format(raw_data(result)[0], raw_data(result)[1])
    print("Encrypted result:\n", enc_result)
    dec_result = result.decrypt(sk).tolist()
    print("Decryption:\n{}\n{}".format(dec_result[0], dec_result[1]))
    plain_mtx = sub(input1, input2)
    print("Plain calculation:\n{}\n{}".format(plain_mtx[0], plain_mtx[1]))
    print("\n")
    print("Dot product\n")
    result = enc_input1 * enc_input2
    enc_result = "{}\n{}".format(raw_data(result)[0], raw_data(result)[1])
    print("Encrypted result:\n", enc_result)
    dec_result = result.decrypt(sk).tolist()
    print("Decryption:\n{}\n{}".format(dec_result[0], dec_result[1]))
    plain_mtx = mul(input1, input2)
    print("Plain calculation:\n{}\n{}".format(plain_mtx[0], plain_mtx[1]))
    print("\n")
    print("Cross product\n")
    result = enc_input1.mm(enc_input2)
    enc_result = "{}\n{}".format(raw_data(result)[0], raw_data(result)[1])
    print("Encrypted result:\n", enc_result)
    dec_result = result.decrypt(sk).tolist()
    print("Decryption:\n{}\n{}".format(dec_result[0], dec_result[1]))
    plain_mtx = matmul(input1, input2)
    print("Plain calculation:\n{}\n{}".format(plain_mtx[0], plain_mtx[1]))
    print("\n")

def circuit(input1, input2):
    vec_mid1 = add(input1, [3.0, 1.0, 4.0])
    vec_mid2 = mul(input2, [1.0, 5.0, 9.0])
    vec_final = sub(vec_mid1, vec_mid2)
    return vec_final

def enc_circuit(input1, input2):
    vec_mid1 = input1 + [3.0, 1.0, 4.0]
    vec_mid2 = input2 * [1.0, 5.0, 9.0]
    vec_final = vec_mid1 - vec_mid2
    return vec_final

def demo4(input1, input2, encryptor, sk):
    print("\033[31mKey decryption difference\033[0m")
    print("1st input vector:\n", input1)
    print("2nd input vector:\n", input2)
    
    print("Plain circuit evaluation:", circuit(input1, input2))
    result = enc_circuit(encryptor.encrypt(input1), encryptor.encrypt(input2))
    print("HE circuit decrypted evaluation:", result.decrypt(sk).tolist())

    bits_scale = 26
    encryptor_tmp = Encryptor(None, 8192, [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31])
    print("Evaluation decrypt with wrong key:", result.decrypt(encryptor_tmp.get_sk()).tolist())



# Create the singleton encoder instance
def main():
    bits_scale = 26
    encryptor = Encryptor(None, 8192, [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31])

    # Encode some data
    vec1 = [5.0, 3.0, 2.0]
    vec2 = [3.0, 2.0, 1.0]
    mtx1 = [[3.0, 3.0], [5.0, 6.0]]
    mtx2 = [[1.0, 2.0], [2.0, 1.0]]

    #encrypt data
    sk = encryptor.get_sk()
    encryptor.make_public()
    
    demo1(vec1, encryptor, sk)
    demo2(vec1, vec2, encryptor, sk)
    demo3(mtx1, mtx2, encryptor, sk)
    demo4(vec1, vec2, encryptor, sk)

if __name__ == "__main__":
    main()

