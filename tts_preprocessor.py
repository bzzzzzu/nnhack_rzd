import re

from num2words import num2words

punctuation = r'[\s,.?!/)\'\]>]'
alphabet_map = {
    "A": " Эй ",
    "B": " Би ",
    "C": " Си ",
    "D": " Ди ",
    "E": " И ",
    "F": " Эф ",
    "G": " Джи ",
    "H": " Эйч ",
    "I": " Ай ",
    "J": " Джей ",
    "K": " Кей ",
    "L": " Эл ",
    "M": " Эм ",
    "N": " Эн ",
    "O": " Оу ",
    "P": " Пи ",
    "Q": " Кью ",
    "R": " Ар ",
    "S": " Эс ",
    "T": " Ти ",
    "U": " Ю ",
    "V": " Ви ",
    "W": " Дабл ю ",
    "X": " Икс ",
    "Y": " Игрик ",
    "Z": " Зэт ",
    "Б": "Бэ",
    'В': "Вэ",
    'Г': 'Гэ',
    'Д': 'Дэ',
    'Ж': 'Жэ',
    'З': 'Зэ',
    'Й': 'Йэ',
    'К': 'Кэ',
    'Л': 'Эл',
    'М': 'Эм',
    'Н': 'Эн',
    'П': 'Пэ',
    'Р': 'Эр',
    'С': 'Эс',
    'Т': 'Тэ',
    'Ф': 'Эф',
    'Х': 'Хэ',
    'Ц': 'Цэ',
    'Ч': 'Чэ',
    'Ш': 'Эш',
    'Щ': 'Эщ'
}


def preprocess(string):
    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('\u201D', '').replace('\u201C', '')  # right and left quote
    string = string.replace('\u201F', '')  # italic looking quote
    string = string.replace('\n', ' ')
    string = convert_num_locale(string)
    string = replace_negative(string)
    string = replace_roman(string)
    string = hyphen_range_to(string)
    
    string = separate_abbreviation_digit(string)
    string = separate_digit_abbreviation(string)
    string = odin(string)
    string = remove_multiple_dots(string)
    string = num_to_words(string)

    string = replace_abbreviations(string)
    string = replace_lowercase_abbreviations(string)

    string = re.sub(rf'\s+({punctuation})', r'\1', string)
    string = string.strip()
    # compact whitespace
    string = ' '.join(string.split())

    return string


def remove_surrounded_chars(string):
    if re.search(r'(?<=alt=)(.*)(?=style=)', string, re.DOTALL):
        m = re.search(r'(?<=alt=)(.*)(?=style=)', string, re.DOTALL)
        string = m.group(0)
    return re.sub(r'\*[^*]*?(\*|$)', '', string)


def convert_num_locale(text):
    pattern = re.compile(r'(?:\s|^)\d{1,3}(?:\.\d{3})+(,\d+)(?:\s|$)')
    result = text
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + result[start:end].replace('.', '').replace(',', '.') + result[end:len(result)]

    pattern = re.compile(r'(\d),(\d)')
    result = pattern.sub(r'\1\2', result)

    return result


def replace_negative(string):
    return re.sub(rf'(\s)(-)(\d+)({punctuation})', r'\1negative \3\4', string)


def replace_roman(string):
    pattern = re.compile(rf'\s[IVXLCDM]{{2,}}{punctuation}')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start + 1] + str(roman_to_int(result[start + 1:end - 1])) + result[end - 1:len(result)]

    return result


def roman_to_int(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


def hyphen_range_to(text):
    pattern = re.compile(r'(\d+)[-–](\d+)')
    result = pattern.sub(lambda x: x.group(1) + ' to ' + x.group(2), text)
    return result


def num_to_words(text):
    # pattern = re.compile(r'\d+\.\d+|\d+')
    pattern = re.compile(r"\d+")
    # result = pattern.sub(lambda x: num2words(float(x.group())), text)
    result = pattern.sub(lambda x: num2words(int(float(x.group())),lang='ru'), text)
    return result


def replace_abbreviations(sequence_2):
    abbreviations_2 = re.findall(r'\b[А-ЯA-Z]{2,}\b', sequence_2)
    for abb in abbreviations_2:
        ch = ''
        for i in abb:
            ch += match_mapping(i)
        sequence_2 = sequence_2.replace(abb, ch)


    return sequence_2


def replace_lowercase_abbreviations(string):
    # abbreviations 1 to 4 characters long, separated by dots i.e. e.g.
    pattern = re.compile(rf'(^|[\s(.\'\[<])(([А-ЯA-Z]\.){{1,4}})({punctuation}|$)')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + replace_abbreviation(result[start:end].upper()) + result[end:len(result)]

    return result


def replace_abbreviation(string):
    result = ""
    for char in string:
        result += match_mapping(char)
    return result


def match_mapping(char):
    for mapping in alphabet_map.keys():
        if char == mapping:
            return alphabet_map[char]
    return char


def separate_abbreviation_digit(text):
    pattern = re.compile(r'(\b[А-ЯA-Z]+)(\d+\b)')
    result = pattern.sub(r'\1 \2', text)
    return result

def separate_digit_abbreviation(text):
    pattern = re.compile(r'(\d+\b)(\b[А-ЯA-Z]+)')
    result = pattern.sub(r'\1 \2', text)
    return result

def remove_multiple_dots(text):
    result = re.sub(r'\.{2,3}', '', text)
    return result

odin_list = ['1', '1.', 'Один', 'один']
def odin(text: str):
    for i in odin_list:
        if i in text: 
            text = text.replace(i, 'од+ин')
    return text

def __main__(args):
    print(preprocess(args[1]))


if __name__ == "__main__":
    import sys
    __main__(sys.argv)
