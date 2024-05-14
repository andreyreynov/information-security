alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
sorted_alphabet = [*alphabet]

message = 'This is an example! I just made up ABC!'
split_message = [*message.upper()]

key = 3

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
print('\n\n')
print(result)
