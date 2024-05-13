import numpy as np

# Зашифрованный текст в виде комбинированной матрицы
combined_matrix = np.array([['П', 'П', 'Е', 'Е', 'П'],
                            ['О', 'Р', 'И', 'И', 'А'],
                            ['В', 'С', '', 'Р', 'В'],
                            ['А', 'Е', 'Ь', 'Н', 'Т'],
                            ['Н', 'Ь', 'А', 'М', 'У']], dtype='str')

# Ключ для расшифровки
key = np.array([[0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0]], dtype='str')


def rotate_and_decrypt(matrix, key, angle):
    rotated_matrix = np.rot90(matrix, angle // 90)
    decrypted_matrix = np.where(key == '1', rotated_matrix, '-')
    return decrypted_matrix


# Итерация по углам поворота
rotate0 = np.rot90(rotate_and_decrypt(combined_matrix, key, 0), 0)
rotate90 = np.rot90(rotate_and_decrypt(combined_matrix, key, 90), 3)
rotate180 = np.rot90(rotate_and_decrypt(combined_matrix, key, 180), 2)
rotate270 = np.rot90(rotate_and_decrypt(combined_matrix, key, 270), 1)

# Извлечение текста из каждой матрицы и объединение его вместе
text_rotate0 = ''.join(rotate0[rotate0 != '-'])
text_rotate90 = ''.join(rotate90[rotate90 != '-'])
text_rotate180 = ''.join(rotate180[rotate180 != '-'])
text_rotate270 = ''.join(rotate270[rotate270 != '-'])

# Соединение текста из всех матриц
decrypted_text = text_rotate0 + text_rotate90 + text_rotate180 + text_rotate270

print('Decrypted Text:', decrypted_text)
