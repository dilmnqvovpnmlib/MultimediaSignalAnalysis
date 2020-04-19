import numpy as np

twice = lambda x: x * 2

if __name__ == '__main__':
    array = np.array([[[1 * i, 2 * i, 3 * i] for i in range(3)] for _ in range(2)])
    print('-'*10)
    array1 = np.array([[[1 * i, 2 * i, 3 * i] for i in range(3)] for _ in range(2)])
    array += array1
    print(array)
    print((array + array1) / 2)
    print('-'*10)
    # print(array.transpose(2, 1, 0))
