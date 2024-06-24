import numpy as np

my_dict = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
print(my_dict)

my_array = np.array(list(my_dict.values()))
print(my_array.shape)
print(type(my_array))
print(my_array)