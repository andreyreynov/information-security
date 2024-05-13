import numpy as np

# Создание шаблонной-матрицы (0 градусов)
matrix = np.array([[0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0],
                   [1, 0, 1, 0, 0]], dtype='str')

# Заменяю '0' на '-', т.к. было лень делать это вручную
matrix[matrix == '0'] = '-'

rotate0 = matrix.copy()
rotate90 = np.rot90(matrix, 3).copy()
rotate180 = np.rot90(matrix, 2).copy()
rotate270 = np.rot90(matrix, 1).copy()

# Текста для вставки
text = 'ПЕРЕНАПРАВЬТЕПИСЬМОИВАНУ'

text_split = [text[i:i + len(text) // 4]
              for i in range(0, len(text), len(text) // 4)]
print('Text parts:', text_split)


def matrix0():
    text_index = 0
    for i in range(rotate0.shape[0]):
        for j in range(rotate0.shape[1]):
            if rotate0[i, j] == '1' and text_index < len(text_split[0]):
                rotate0[i, j] = text_split[0][text_index]
                text_index += 1

    print('Rotate 0:\n', rotate0)


def matrix90():
    text_index = 0
    for i in range(rotate90.shape[0]):
        for j in range(rotate90.shape[1]):
            if rotate90[i, j] == '1' and text_index < len(text_split[1]):
                rotate90[i, j] = text_split[1][text_index]
                text_index += 1

    print('Rotate 90:\n', rotate90)


def matrix180():
    text_index = 0
    for i in range(rotate180.shape[0]):
        for j in range(rotate180.shape[1]):
            if rotate180[i, j] == '1' and text_index < len(text_split[2]):
                rotate180[i, j] = text_split[2][text_index]
                text_index += 1

    print('Rotate 180:\n', rotate180)


def matrix270():
    text_index = 0
    for i in range(rotate270.shape[0]):
        for j in range(rotate270.shape[1]):
            if rotate270[i, j] == '1' and text_index < len(text_split[3]):
                rotate270[i, j] = text_split[3][text_index]
                text_index += 1

    print('Rotate 270:\n', rotate270)


matrix0()
matrix90()
matrix180()
matrix270()
