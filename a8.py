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

# xor_data = [
#     ([0, 0], [0]),
#     ([0, 1], [1]),
#     ([1, 0], [1]),
#     ([1, 1], [0]),
# ]
# xor_nn = NeuralNet(2, 2, 1)
# xor_nn.train(xor_data, iters = 10000, print_interval= 1000)
# 
# print(xor_nn.test_with_expected(xor_data))
# print(xor_nn.evaluate([1, 1]))

votr_data = [
    ([.9, .6, .8, .3, .1], [1]),
    ([.8, .8, .4, .6, .4], [1]),
    ([.7, .2, .4, .6, .3], [1]),
    ([.5, .5, .8, .4, .8], [0]),
    ([.3, .1, .6, .8, .8], [0]),
    ([.6, .3, .4, .3, .6], [0])
]

voter_nn = NeuralNet(5, 6, 1)
voter_nn.train(votr_data)
print(voter_nn.test_with_expected(votr_data))
print(voter_nn.test([
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6],
]))
print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")



