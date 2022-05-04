import numpy as np

arr = np.random.randint(5, size=(2, 4))
print("Array = \n", arr)
print(np.unique(arr, return_counts=True))
