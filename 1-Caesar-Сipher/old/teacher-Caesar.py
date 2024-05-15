test = """
AZ  0   AZ
AZ  1   BA
AZ  3   DC
"""[1:-1].split('\n')

msg = 'AZ'

abc = [chr(val) for val in range(ord('A'), ord('A') + 26)]


def encode_caesar(msg, key):
    msg = msg.upper()
    out_msg = ''
    for char in msg:
        old_code = abc.index(char)
        new_code = (old_code + key) % 26
        out_msg += abc[new_code]
    return out_msg


for _ in test:
    msg, key, truth = _.split()
    key = int(key)
    out = encode_caesar(msg, key)
    print(msg, key, truth, sep='\t', end='\t')
    print(out, out == truth, sep='\t', end='\t')
    print()


msg = 'Hello'


def is_real_word(word):
    return word == msg.upper()


def decode_bruteforce(msg):
    for key in range(26):
        decoded_word = encode_caesar(msg, -key)
        if is_real_word(decoded_word):
            return key, decoded_word
    return None, msg

key, _ = decode_bruteforce(encode_caesar(msg, 4))
print(key, _)

