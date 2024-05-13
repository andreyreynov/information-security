import numpy as np

# Создание шаблонной-матрицы (0 градусов)
matrix = np.array([[0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0],
                   [1, 0, 1, 0, 0]], dtype='str')

# Заменяем '0' на '-', чтобы улучшить визуальное отображение
matrix[matrix == '0'] = '-'

rotate0 = matrix.copy()
rotate90 = np.rot90(matrix, 3).copy()
rotate180 = np.rot90(matrix, 2).copy()
rotate270 = np.rot90(matrix, 1).copy()

# Текст для вставки
text = 'ПЕРЕНАПРАВЬТЕПИСЬМОИВАНУ'

text_split = [text[i:i + len(text) // 4]
              for i in range(0, len(text), len(text) // 4)]
print('Text parts:', text_split)


def insert_text_into_matrix(matrix, text_part):
    text_index = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == '1' and text_index < len(text_part):
                matrix[i, j] = text_part[text_index]
                text_index += 1

    return matrix


def combine_matrices(*matrices):
    combined_matrix = np.empty_like(matrices[0], dtype='str')
    for matrix in matrices:
        combined_matrix[matrix != '-'] = matrix[matrix != '-']
    return combined_matrix


rotate_matrices = [rotate0, rotate90, rotate180, rotate270]

for rotate_matrix in rotate_matrices:
    insert_text_into_matrix(rotate_matrix, text_split.pop(0))

result_matrix = combine_matrices(rotate0, rotate90, rotate180, rotate270)

print('Combined Matrix:')
print(result_matrix)
