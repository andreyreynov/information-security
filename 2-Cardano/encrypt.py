import numpy as np

# Создание шаблонной-матрицы (0 градусов)
key = np.array([[0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0]], dtype='str')

# Заменяем '0' на '-', чтобы улучшить визуальное отображение
key[key == '0'] = '-'

rotate0 = key.copy()
rotate90 = np.rot90(key, 3).copy()
rotate180 = np.rot90(key, 2).copy()
rotate270 = np.rot90(key, 1).copy()

# Текст для вставки
text = 'ПЕРЕНАПРАВЬТЕПИСЬМОИВАНУ'

text_split = [text[i:i + len(text) // 4]
              for i in range(0, len(text), len(text) // 4)]
print('Text parts:', text_split)


def insert_text_into_key(key, text_part):
    text_index = 0
    for i in range(key.shape[0]):
        for j in range(key.shape[1]):
            if key[i, j] == '1' and text_index < len(text_part):
                key[i, j] = text_part[text_index]
                text_index += 1

    return key


def combine_matrices(*matrices):
    combined_key = np.empty_like(matrices[0], dtype='str')
    for key in matrices:
        combined_key[key != '-'] = key[key != '-']
    return combined_key


rotate_matrices = [rotate0, rotate90, rotate180, rotate270]

for rotate_key in rotate_matrices:
    insert_text_into_key(rotate_key, text_split.pop(0))

combined_matrix = combine_matrices(rotate0, rotate90, rotate180, rotate270)

print('Combined key:')
print(combined_matrix)
