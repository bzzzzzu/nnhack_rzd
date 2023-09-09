import os.path

def get_text_from_document(path):
    with open(path, 'r', encoding='utf-8') as f:
    #with open(path, 'r', errors='ignore') as f:
        return f.readlines()

if not os.path.exists('data-processing/tables/'):
    os.makedirs('data-processing/tables/')
if not os.path.exists('data-processing/tables_fixed/'):
    os.makedirs('data-processing/tables_fixed/')

csv_list = get_text_from_document('data-processing/test2.csv')
num_list = 0
start_parse = 0
curr_list = []
curr_fixed_list = []
curr_col0 = ''
curr_col1 = ''
curr_col2 = ''
curr_col3 = ''
for c in csv_list:
    if start_parse == 1:
        if 'и устранению неисправностей' in c:
            print(c)
            print(len(curr_list))
            if len(curr_list) > 0:
                with open(f'data-processing/tables/table_{num_list}.csv', 'w', encoding='utf-8') as f:
                    for cl in curr_list:
                        f.write(cl)
                with open(f'data-processing/tables_fixed/table_{num_list}.csv', 'w', encoding='utf-8') as f:
                    for cl in curr_fixed_list:
                        f.write(cl + '\n')
                curr_col0 = ''
                curr_col1 = ''
                curr_col2 = ''
                curr_col3 = ''
                print('-------------------')
            curr_list = []
            curr_fixed_list = []
            num_list = num_list + 1
        c2 = str.replace(c, '\n', '')
        csv_str = str.split(c2, ';')
        '''
        if len(csv_str) > 3:
            if csv_str[1] == '' and csv_str[2] == '' and csv_str[3] == '' and len(c2) > 10 and not csv_str[0].isnumeric():
                print(c2)
            if csv_str[1] == '' and csv_str[2] == '' and len(c2) > 10:
                print(c2)
        '''
        #if c.count(';;;') > 0 and len(c) > 10:
        #    print(c)
        if len(csv_str) > 3:
            write_row = 1
            fixed_str = ''
            if csv_str[0] != '':
                if csv_str[0].isnumeric():
                    curr_col1 = csv_str[0]
                    fixed_str = curr_col0 + ';' + csv_str[0]
                else:
                    curr_col0 = csv_str[0]
                    write_row = 0
                    #fixed_str = csv_str[0] + ';' + curr_col1
            else:
                fixed_str = curr_col0 + ';' + curr_col1
            fixed_str = fixed_str + ';'

            if csv_str[1] != '':
                fixed_str = fixed_str + csv_str[1]
                curr_col2 = csv_str[1]
            else:
                fixed_str = fixed_str + curr_col2
            fixed_str = fixed_str + ';'

            if csv_str[2] != '':
                fixed_str = fixed_str + csv_str[2]
                curr_col3 = csv_str[2]
            else:
                fixed_str = fixed_str + curr_col3
            fixed_str = fixed_str + ';'
            if write_row == 1:
                if len(curr_fixed_list) > 0:
                    if curr_fixed_list[-1] != fixed_str:
                        curr_fixed_list.append(fixed_str)
                else:
                    curr_fixed_list.append(fixed_str)

        curr_list.append(c)
    if 'Приложение N 40 к настоящему Перечню' in c:
        start_parse = 1
if len(curr_list) > 0:
    with open(f'data-processing/tables/table_{num_list}.csv', 'w', encoding='utf-8') as f:
        for cl in curr_list:
            f.write(cl)
    with open(f'data-processing/tables_fixed/table_{num_list}.csv', 'w', encoding='utf-8') as f:
        for cl in curr_fixed_list:
            f.write(cl + '\n')

print(len(csv_list))