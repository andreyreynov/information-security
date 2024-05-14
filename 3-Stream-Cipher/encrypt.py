import random


message = 'GOOGLE WOW COOL'

# Преобразование текста в бинарную последовательность
bit_message = ''.join(format(ord(x), '08b') for x in message)


def generate_initial_state(N):
    initial_state = ""
    for i in range(N):
        x = num = random.randint(0, 1)
        initial_state += str(x)
    return initial_state


def generate_R1():
    R1_initial_state = generate_initial_state(19)
    R1_list = []

    for bit in bit_message:
        num_18 = int(R1_initial_state[18])
        num_17 = int(R1_initial_state[17])
        num_16 = int(R1_initial_state[16])
        num_13 = int(R1_initial_state[13])

        xor_X = num_18 ^ num_17
        xor_Y = xor_X ^ num_16
        xor_Z = xor_Y ^ num_13

        R1_list.append(xor_Z)

        # Сдвигаем регистр на один символ влево
        R1_initial_state = str(xor_Z) + R1_initial_state[:-1]

        # Добавляем синхробит
        R1_initial_state += str(xor_Z)

    return R1_list


def generate_R2():
    R2_initial_state = generate_initial_state(23)
    R2_list = []

    for bit in bit_message:
        num_18 = int(R2_initial_state[18])
        num_17 = int(R2_initial_state[17])

        xor_X = num_18 ^ num_17

        R2_list.append(xor_X)

        # Сдвигаем регистр на один символ влево
        R2_initial_state = str(xor_X) + R2_initial_state[:-1]

        # Добавляем синхробит
        R2_initial_state += str(xor_X)

    return R2_list


def generate_R3():
    R3_initial_state = generate_initial_state(25)
    R3_list = []

    for bit in bit_message:
        num_22 = int(R3_initial_state[22])
        num_21 = int(R3_initial_state[21])
        num_20 = int(R3_initial_state[20])
        num_7 = int(R3_initial_state[7])

        xor_X = num_22 ^ num_21
        xor_Y = xor_X ^ num_20
        xor_Z = xor_Y ^ num_7

        R3_list.append(xor_Z)

        # Сдвигаем регистр на один символ влево
        R3_initial_state = str(xor_Z) + R3_initial_state[:-1]

        # Добавляем синхробит
        R3_initial_state += str(xor_Z)

    return R3_list


def generate_key(R1_list, R2_list, R3_list):
    R12_list = []
    key = []

    # Взять каждый символ R1_list и R2_list
    for i in range(len(R1_list)):
        xor_R1_R2 = R1_list[i] ^ R2_list[i]
        R12_list.append(xor_R1_R2)

    # Взять R12_list и R3_list
    for i in range(len(R12_list)):
        xor_R12_R3 = R12_list[i] ^ R3_list[i]
        key.append(xor_R12_R3)

    return key


def list_to_bits(lst):
    return ''.join(map(str, lst))


# Сложение двух битовых строк
def add_bits(bit_str1, bit_str2):
    result = ''
    for bit1, bit2 in zip(bit_str1, bit_str2):
        result += str(int(bit1) ^ int(bit2))
    return result


# Генерация последовательностей
R1_list = generate_R1()
R2_list = generate_R2()
R3_list = generate_R3()

# Генерация ключа за счет R1, R2, R3
key = generate_key(R1_list, R2_list, R3_list)

# Сложение бинарных строк
summed_bits = add_bits(bit_message, list_to_bits(key))

# Вывод результирующей строки
print(
    "\n\tThe generated key and the encrypted message are saved in \033[4mfile.txt\033[0m.\n\tUse \033[4mdecrypt.py\033[0m to decrypt the message.\n")

with open('file.txt', 'w', encoding='utf-8') as f:
    f.write('Original message in bits:\n')
    f.write(f'{bit_message}\n\n')
    f.write('Key:\n')
    f.write(f'{list_to_bits(key)}\n\n')
    f.write('After combining together we get these bits:\n')
    f.write(f'{summed_bits}\n\n')
