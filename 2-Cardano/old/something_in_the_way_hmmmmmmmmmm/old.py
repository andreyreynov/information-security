# NumPy rotation
# https://numpy.org/doc/stable/reference/generated/numpy.rot90.html

# import pymorphy2
import numpy as np

# Создание шаблонной-матрицы (0 градусов)
matrix = np.matrix("0 1 0 1 0; 0 0 0 0 0; 0 0 0 1 0; 0 1 0 0 0; 1 0 1 0 0")
print('\nOriginal matrix: \n', matrix)

# Поворот матрицы-шаблона по часовой стрелке
rotate90 = np.rot90(matrix, 3)
print('\n90 degrees rotation:\n', rotate90)
rotate180 = np.rot90(matrix, 2)
print('\n180 degrees rotation:\n', rotate180)
rotate270 = np.rot90(matrix, 1)
print('\n270 degrees rotation:\n', rotate270)

# Count amount of 1's
oneCount = []
matrix0 = oneCount.append(np.count_nonzero(matrix == 1))
matrix90 = oneCount.append(np.count_nonzero(rotate90 == 1))
matrix180 = oneCount.append(np.count_nonzero(rotate180 == 1))
matrix270 = oneCount.append(np.count_nonzero(rotate270 == 1))
oneCount = sum(oneCount)

text = 'ВОТТАКОЙТОСЛУЧАЙНЫЙТЕКСТ'
text = [*text]

if len(text) <= oneCount:
    print('you can continue coding')
else:
    print('wow you suck!')
