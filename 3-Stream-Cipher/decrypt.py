encrypted_message = '000100101011001101001000010111111001101010001000111001101001110101110100001011011100010011101100110000001011011000011001'

key = '010101011111110000000111000110001101011011001101110001101100101000111011011110101110010010101111100011111111100101010101'


def subtract_bits(encrypted_text, key):
    # This function substacts the key from the encrypted message
    result = ''
    for bit1, bit2 in zip(encrypted_text, key):
        result += str(int(bit1) ^ int(bit2))
    return result


# Original text without a key applied (in bits format)
bit_message = subtract_bits(encrypted_message, key)

# Here we turn the original text (in bits format) into a text
message = ''.join(
    chr(int(bit_message[i:i+8], 2)) for i in range(0, len(bit_message), 8))

print(f'Regular message:\n\n\t{message}\n')
