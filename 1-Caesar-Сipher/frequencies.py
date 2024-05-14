from collections import Counter
import string


class EnglishFrequecies:
    def __init__(self):
        '''
        Contains the frequencies of the English alphabet.

        Example:
        freq = EnglishFrequecies()
        print(freq.letters)

        '''

        self.letters = {

            'A':	8.34,
            'B':	1.54,
            'C':	2.73,
            'D':	4.14,
            'E':	12.60,
            'F':	2.03,
            'G':	1.92,
            'H':	6.11,
            'I':	6.71,
            'J':	0.23,
            'K':	0.87,
            'L':	4.24,
            'M':	2.53,
            'N':	6.80,
            'O':	7.70,
            'P':	1.66,
            'Q':	0.09,
            'R':	5.68,
            'S':	6.11,
            'T':	9.37,
            'U':	2.85,
            'V':	1.06,
            'W':	2.34,
            'X':	0.20,
            'Y':	2.04,
            'Z':	0.06,

        }
        self.letters_sorted = dict(sorted(
            self.letters.items(), key=lambda x: x[1], reverse=True))

        self.digram = {

        }


class RussianFrequencies:
    def __init__(self):
        '''
        Contains the frequencies of the Russian alphabet.

        Example:
        freq = RussianFrequencies()
        print(freq.letters)

        '''

        self.letters = {
            'А':	7.64,
            'Б':	2.01,
            'В':	4.38,
            'Г':	1.72,
            'Д':	3.09,
            'Е':	8.75,
            'Ё':	0.20,
            'Ж':	1.01,
            'З':	1.48,
            'И':	7.09,
            'Й':	1.21,
            'К':	3.30,
            'Л':	4.96,
            'М':	3.17,
            'Н':	6.78,
            'О':	11.18,
            'П':	2.47,
            'Р':	4.23,
            'С':	4.97,
            'Т':	6.09,
            'У':	2.22,
            'Ф':	0.21,
            'Х':	0.95,
            'Ц':	0.39,
            'Ч':	1.40,
            'Ш':	0.72,
            'Щ':	0.30,
            'Ъ':	0.02,
            'Ы':	2.36,
            'Ь':	1.84,
            'Э':	0.36,
            'Ю':	0.47,
            'Я':	1.9
        }

        self.letters_sorted = dict(sorted(
            self.letters.items(), key=lambda x: x[1], reverse=True))

        self.digram = {

            'АЛ':	0.97,
            'АН':	0.80,
            'БЫ':	0.85,
            'ВЕ':	0.77,
            'ВО':	0.89,
            'ГО':	0.99,
            'ДЕ':	0.77,
            'ЕЛ':	0.84,
            'ЕН':	1.22,
            'ЕР':	1.00,
            'ЕТ':	0.82,
            'КА':	0.87,
            'КО':	1.25,
            'ЛА':	0.80,
            'ЛИ':	1.12,
            'ЛО':	0.76,
            'ЛЬ':	0.76,
            'НА':	1.42,
            'НЕ':	1.23,
            'НИ':	1.25,
            'НО':	1.46,
            'ОВ':	0.93,
            'ОЛ':	0.99,
            'ОН':	1.06,
            'ОР':	0.79,
            'ОС':	0.82,
            'ОТ':	0.93,
            'ПО':	1.16,
            'ПР':	0.87,
            'РА':	1.13,
            'РЕ':	0.89,
            'РО':	1.00,
            'СТ':	1.55,
            'ТА':	0.87,
            'ТЕ':	0.75,
            'ТО':	1.72,
            'ТЬ':	0.89
        }

        self.digram_sorted = dict(
            sorted(self.digram.items(), key=lambda x: x[1], reverse=True))

        self.trigram = {
            'АТЬ':	0.42,
            'БЫЛ':	0.66,
            'ВЕР':	0.38,
            'ЕГО':	0.42,
            'ЕНИ':	0.46,
            'ЕНН':	0.32,
            'ЕСТ':	0.34,
            'КАК':	0.30,
            'ЛЬН':	0.31,
            'ОВА':	0.31,
            'ОГО':	0.37,
            'ОЛЬ':	0.43,
            'ОРО':	0.30,
            'ОСТ':	0.45,
            'ОТО':	0.33,
            'ПРИ':	0.35,
            'ПРО':	0.38,
            'СТА':	0.35,
            'СТВ':	0.38,
            'ТОР':	0.34,
            'ЧТО':	0.58,
            'ЭТО':	0.36
        }

        self.trigram_sorted = dict(
            sorted(self.trigram.items(), key=lambda x: x[1], reverse=True))


class GetFrequencies():
    '''
    Get the frequencies of a message.

    Example:
    get_freq = GetFrequencies()
    print(get_freq.get_freqs(message))
    '''

    def get_freqs(self, message):
        '''
        Prints the 10 most common characters in the message.

        Args:
        message (str): The message to be analyzed.

        Example:
        print(get_freq.get_freqs(message))
        '''
        # Удаляем знаки препинания и пробелы из сообщения
        message = ''.join(
            char for char in message if char not in string.punctuation + ' ')
        return Counter(message).most_common(30)

    def get_freqs_percentage(self, message):
        # Удаляем знаки препинания и пробелы из сообщения
        message = ''.join(
            char for char in message if char not in string.punctuation + ' ')
        freqs = Counter(message).most_common(30)
        total_elements = sum(count for _, count in freqs)
        freqs_with_percentage = [
            (char, round(count / total_elements * 100, 3)) for char, count in freqs]
        return freqs_with_percentage

    def get_digram(self, message, syllable_length=2):
        syllables = []

        for i in range(len(message) - syllable_length + 1):
            current_syllable = message[i:i+syllable_length]
            syllables.append(current_syllable)

        counter = Counter(syllables)
        # Удаляем пробелы из результатов и игнорируем одиночные буквы
        counter = {key.replace(' ', ''): value for key, value in counter.items() if len(
            key.strip()) == syllable_length}
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    def get_digram_percentage(self, message, syllable_length=2):
        syllables = []

        for i in range(len(message) - syllable_length + 1):
            current_syllable = message[i:i+syllable_length]
            syllables.append(current_syllable)

        counter = Counter(syllables)
        total_syllables = sum(counter.values())
        # Вычисляем процент повторений для каждой пары букв
        digram_with_percentage = {key.replace(' ', ''): round(value / total_syllables * 100, 3)
                                  for key, value in counter.items() if len(key.strip()) == syllable_length}
        return dict(sorted(digram_with_percentage.items(), key=lambda x: x[1], reverse=True))
