import numpy as np

def suma_arrays(lista_1=[1,2,3,4,5], lista_2=[6,7,8,9,10]):
    array_1 = np.array(lista_1)
    array_2 = np.array(lista_2)
    return array_1 + array_2

if __name__ == '__main__':
    print(suma_arrays())