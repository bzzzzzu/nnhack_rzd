import pandas as pd
import os
import sys
import shutil


path_to_folder = './extracted_tables/'
path_to_kasha_csv = 'test2.csv'

def main():
    remove_old_and_create_new_folder(path_to_folder=path_to_folder)
    extract_tables_as_csv_files_in_folder(path_to_kasha_csv=path_to_kasha_csv)


def remove_old_and_create_new_folder(path_to_folder):
    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(path_to_folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    os.makedirs(path_to_folder)

def extract_tables_as_csv_files_in_folder(path_to_kasha_csv):
    prev_row = ''
    table_counter = 1
    flag = False
    prev_row = ''
    with open(path_to_kasha_csv, 'r', encoding='utf-8') as kasha:
        for row in kasha.readlines():

            # Find start of table
            if row == 'N";Неисправность;Вероятная причина;Метод устранения\n' or row == 'N;Неисправность;Вероятная причина;Метод устранения неисправности\n':
                flag = True
                continue

            # Find end of table
            if flag == True and row == ';;;\n' and prev_row == ';;;\n':
                flag = False
                table_counter += 1

            # Write table row by row to new table file (for each table separate file)
            if flag == True:
                with open(f'./extracted_tables/extracted_table_{table_counter}.csv', 'a', encoding='utf-8') as new_table_file:
                    new_table_file.write(row)


            prev_row = row


if __name__ == '__main__':
    main()