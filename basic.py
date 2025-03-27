inputs = [1, 2, 3, 4]

weights = [[0.1, 0.6, 1, 2, 0.8], [0.8, 1, 0.5, 0.4, 0.36], [0.2, 2, 0.69, 0.38, 0.14], [0.33, 1.65, 0.56, 0.91, 0.13]]

biases = [1, 3.2, 0.54, 2, 3.5]

output = []

for n_weights, n_biases in zip(weights, biases):
    n_output = 0
    for n_input, n_weight in zip(inputs, n_weights):
        n_output += n_input * n_weight
        n_output += n_biases
        output.append(n_output)
        
print(output)
        
        

