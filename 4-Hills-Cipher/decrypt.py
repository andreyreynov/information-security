import numpy as np


def mod_inverse(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None


def matrix_mod_inverse(matrix, modulus):
    det = int(np.round(np.linalg.det(matrix)))
    det_inv = mod_inverse(det % modulus, modulus)

    if det_inv is None:
        raise ValueError("Обратная матрица не существует")

    matrix_modulus_inv = det_inv * \
        np.round(det * np.linalg.inv(matrix)).astype(int) % modulus
    return matrix_modulus_inv


def hill_cipher_decrypt(cipher_text, key_matrix):
    cipher_text = cipher_text.upper()

    text_vector = [ord(char) - ord('A') for char in cipher_text]
    text_matrix = np.array(text_vector).reshape(-1, 2).T

    key_matrix = np.array(key_matrix)
    key_matrix_inv = matrix_mod_inverse(key_matrix, 26)

    decrypted_matrix = np.dot(key_matrix_inv, text_matrix) % 26
    decrypted_text = ''.join(chr(int(num) + ord('A'))
                             for num in decrypted_matrix.T.flatten())

    return decrypted_text


cipher_text = "GCANZHYN"
key_matrix = [[15, 0], [19, 7]]

decrypted_text = hill_cipher_decrypt(cipher_text, key_matrix)
print(f"Encrypted text: {cipher_text}")
print(f"Decrypted text: {decrypted_text}")
