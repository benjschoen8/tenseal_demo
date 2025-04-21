import tenseal as ts
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from classes import *

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

def circuit(input1, input2, encryptor):
    vec_mid1 = input1 + [3.0, 1.0, 4.0]
    vec_mid2 = input2 * [1.0, 5.0, 9.0]
    vec_final = vec_mid1 - vec_mid2
    print("first iteration: {} + [3.0, 1.0, 4.0] = {}".format(raw_data(input1), raw_data(vec_mid1)))
    print("second iteration: {} + [1.0, 5.0, 9.0] = {}".format(raw_data(input2), raw_data(vec_mid2)))
    print("last iteration: {} + {} = {}".format(raw_data(vec_mid1), raw_data(vec_mid2), raw_data(vec_final)))
    return vec_final

def nn_model(encryptor):
    def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        for data, target in test_loader:
            # Encoding and encryption
            x_enc, windows_nb = ts.im2col_encoding(
                context, data.view(28, 28).tolist(), kernel_shape[0],
                kernel_shape[1], stride
            )
            # Encrypted evaluation
            enc_output = enc_model(x_enc, windows_nb)
            # Decryption of result
            #output = enc_output.decrypt()
            #output = torch.tensor(output).view(1, -1)
            #
            ## compute loss
            #loss = criterion(output, target)
            #test_loss += loss.item()
            #
            ## convert output probabilities to predicted class
            #_, pred = torch.max(output, 1)
            ## compare predictions to true label
            #correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            ## calculate test accuracy for each object class
            #label = target.data[0]
            #class_correct[label] += correct.item()
            #class_total[label] += 1


        # calculate and print avg test loss
        test_loss = test_loss / sum(class_total)
        print(f'Test Loss: {test_loss:.6f}\n')

        for label in range(10):
            print(
                f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
                f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
            )

         
    model = ConvNet()
    model.load_state_dict(torch.load('mnist_convnet.pth'))
    model.eval()

    # Load one element at a time
    criterion = torch.nn.CrossEntropyLoss()
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]
    enc_model = EncConvNet(model)
    enc_test(encryptor.get_context(), enc_model, test_loader, criterion, kernel_shape, stride)

# Create the singleton encoder instance
def main():
    bits_scale = 26
    encryptor = Encryptor(None, 8192, [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31])

    # Encode some data
    vec1 = [5.0, 3.0, 2.0]
    vec2 = [3.0, 2.0, 1.0]
    mtx1 = [[3.0, 3.0], [5.0, 6.0]]
    mtx2 = [[1.0, 2.0], [2.0, 1.0]]

    print("Original data:", vec1)
    print("Original data:", vec2)
    print("Original data:", mtx1)
    print("Original data:", mtx2)

    #encrypt data
    sk = encryptor.get_sk()
    encryptor.make_public()
    
    enc_vec1 = encryptor.encrypt(vec1)
    enc_vec2 = encryptor.encrypt(vec2)
    obj_sum = enc_vec1 + enc_vec2
    obj_sub = enc_vec1 - enc_vec2
    enc_mtx1 = encryptor.encrypt(mtx1)
    enc_mtx2 = encryptor.encrypt(mtx2)
    
    print("Encrypted data:", raw_data(enc_vec1))
    print("Encrypted data:", raw_data(enc_vec2))
    print("Encrypted data:", raw_data(obj_sum))
    print("Encrypted data:", raw_data(obj_sub))
    print("Encrypted data:", raw_data(enc_mtx1))
    print("Encrypted data:", raw_data(enc_mtx2))

    print("Decrypted data:", enc_vec1.decrypt(sk).tolist())
    print("Decrypted data:", enc_vec2.decrypt(sk).tolist())
    print("Decrypted data:", obj_sum.decrypt(sk).tolist())
    print("Decrypted data:", obj_sub.decrypt(sk).tolist())
    print("Decrypted data:", enc_mtx1.decrypt(sk).tolist())
    print("Decrypted data:", enc_mtx2.decrypt(sk).tolist())

    result_circuit = circuit(enc_vec1, enc_vec2, encryptor)
    print("result for circuit:", result_circuit.decrypt(sk).tolist())

    nn_model(encryptor)
    print(
            f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
            f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
            )


    import inspect
    import pprint

    def introspect(obj):
        print("Type:", type(obj))
        print("\nAttributes and Methods (from dir):")
        pprint.pprint(dir(obj))

        try:
            print("\n__dict__:")
            pprint.pprint(obj.__dict__)
        except AttributeError:
            print("\nNo __dict__ available.")

        print("\nInspect (getmembers):")
        for name, value in inspect.getmembers(obj):
            if not name.startswith('__'):
                print(f"{name}: {value}")

# Usage:
    #introspect(dec_vec1.tolist())
    #introspect(enc_vec2)
    #introspect(enc_mtx2)
    #introspect(sum12_vec.decrypt(sk).tolist())

if __name__ == "__main__":
    main()

