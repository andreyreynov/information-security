import numpy as np


def hill_cipher_encrypt(text, key_matrix):

    text = text.replace(" ", "").upper()
    if len(text) % 2 != 0:
        text += 'x'  # Добавляем 'x' если длина текста нечетная

    text_vector = [ord(char) - ord('A') for char in text]
    text_matrix = np.array(text_vector).reshape(-1, 2).T

    key_matrix = np.array(key_matrix)

    encrypted_matrix = np.dot(key_matrix, text_matrix) % 26
    encrypted_text = ''.join(chr(num + ord('A'))
                             for num in encrypted_matrix.T.flatten())

    return encrypted_text


text = "lemonade"
key_matrix = [[3, 6], [4, 7]]

encrypted_text = hill_cipher_encrypt(text, key_matrix)
print(f"Original text: {text}")
print(f"Encrypted text: {encrypted_text}")
