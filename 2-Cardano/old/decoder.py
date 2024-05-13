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


def decrypt(matrix, key):
    decrypted_matrix = np.empty_like(matrix, dtype='U1')
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if key[row, col] == '1':
                decrypted_matrix[row, col] = matrix[row, col]
            else:
                decrypted_matrix[row, col] = '-'
    return decrypted_matrix


# Функция для поворота матрицы на 90 градусов
def rotate_matrix(matrix, angle):
    if angle == 0:
        return matrix
    elif angle == 90:
        return np.rot90(matrix, 1)
    elif angle == 180:
        return np.rot90(matrix, 2)
    elif angle == 270:
        return np.rot90(matrix, 3)
    else:
        print("Invalid angle. Angle must be 0, 90, 180, or 270.")
        return None


def decrypt_and_print(matrix, key):
    decrypted_matrix = decrypt(matrix, key)
    return decrypted_matrix


# Поворачиваем матрицу на 0 градусов
rotated_matrix_0 = rotate_matrix(combined_matrix, 0)
print(f'Rotated Matrix (0 degrees):\n{rotated_matrix_0}\n')
decrypt_and_print(rotated_matrix_0, key)

# Поворачиваем матрицу на 90 градусов
rotated_matrix_90 = rotate_matrix(combined_matrix, 90)
print(f'Rotated Matrix (90 degrees):\n{rotated_matrix_90}\n')
print(np.rot90(decrypt_and_print(rotated_matrix_90, key), 3))

# Поворачиваем матрицу на 180 градусов
rotated_matrix_180 = rotate_matrix(combined_matrix, 180)
print(f'Rotated Matrix (180 degrees):\n{rotated_matrix_180}\n')
print(np.rot90(decrypt_and_print(rotated_matrix_180, key), 2))


# Поворачиваем матрицу на 270 градусов
rotated_matrix_270 = rotate_matrix(combined_matrix, 270)
print(f'Rotated Matrix (270 degrees):\n{np.rot90(rotated_matrix_270, 1)}\n')
print(np.rot90(decrypt_and_print(rotated_matrix_270, key), 1))
