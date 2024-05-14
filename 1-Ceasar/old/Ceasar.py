import pyperclip

message = 'This is an example I just made up ABC'

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def encode():
    print('\n\n\n----Encode----\nВведите текст:')
    message = input()

    sorted_alphabet = [*alphabet]

    split_message = [*message.upper()]

    key = int(input('Введите уровень смещения:'))

    encoded_message = []
    for char in split_message:
        if char == ' ':
            encoded_message.append(' ')
        elif char.isalpha():  # Проверяем, является ли символ буквой
            original_index = sorted_alphabet.index(char)
            new_index = (original_index + key) % len(sorted_alphabet)
            encoded_char = sorted_alphabet[new_index]
            encoded_message.append(encoded_char)
        else:
            encoded_message.append(char)  # Добавляем знаки препинания как есть

    result = ''.join(encoded_message)
    pyperclip.copy(result)
    print('\n\n')
    print(result)

    while True:
        text_to_speak = input('\nВведите команду:')
        if text_to_speak == 'back':
            break
        elif text_to_speak == 'again':
            encode()
        elif text_to_speak == 'decode':
            decode()


def decode():
    print('\n\n\n----Decode----\nВведите текст:')
    message = input()

    for key in range(len(alphabet)):
        translated = ''
        for ch in message:
            if ch in alphabet:
                num = alphabet.find(ch)
                num = num - key
                if num < 0:
                    num = num + len(alphabet)
                translated = translated + alphabet[num]
            else:
                translated = translated + ch
        print('Hacking key is %s: %s' % (key, translated))

    while True:
        text_to_speak = input('\nВведите команду:')
        if text_to_speak == 'back':
            break
        elif text_to_speak == 'again':
            decode()
        elif text_to_speak == 'encode':
            encode()


while True:
    print('1. Encode\n2. Decode')
    text_to_speak = input('Выберите действие (1/2):')

    if text_to_speak == '1':
        encode()
    elif text_to_speak == '2':
        decode()
