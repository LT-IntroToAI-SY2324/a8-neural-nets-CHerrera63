from neural import *
# train_data = [
#     ([0, 0], [0]),
#     ([0, 1], [1]),
#     ([1, 0], [1]),
#     ([1, 1], [0]),
# ]



# print(gtn.evaluate([0, 0]))
# print(gtn.evaluate([0, 1]))
# print(gtn.evaluate([1, 0]))

xor_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]
xor_nn = NeuralNet(2, 2, 1)
xor_nn.train(xor_data, iters = 1000, print_interval= 100)
print(xor_nn.test_with_expected(xor_data))
print(xor_nn.evaluate([1, 1]))
print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")



