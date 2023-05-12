import pandas as pd
from tqdm import tqdm
from datasets import list_datasets, load_dataset
import re

## load dataset
dataset = load_dataset('wikisql')

tables = set()

with open('datasets/english2SQL-processed/eng2SQL_train.txt', 'w') as file:
    for i in range(20):
        file.write("This is input\tThis is output\n")


def construct_names(table_name, table_header, column_index, column_value):
    res = []
    columns = list(zip(column_value, column_index))
    for value, index in columns:
        short_name = table_header[index]
        short_name = short_name.replace(' ', '@')
        full_name = '[' + table_name + ']' + '[' + short_name + ']'
        res.append((full_name, value))
    return res


table_end = re.compile(r' table$')
table_mid = re.compile(r' table ')
train_set = []


def ireplace(old, new, text):
    idx = 0
    while idx < len(text):
        index_l = text.lower().find(old.lower(), idx)
        if index_l == -1:
            return text
        text = text[:index_l] + new + text[index_l + len(old):]
        idx = index_l + len(new)
    return text


with open('datasets/english2SQL-processed/eng2SQL_train.txt', 'w') as file:
    for row in tqdm(dataset['train']):
        tables.add(row['table']['name'])
        question = row['question']
        human_sql = row['sql']['human_readable']
        table_name = row['table']['name']
        if table_name == '':
            table_name = 'Unknown'
        table_header = row['table']['header']
        sql_column_index = row['sql']['conds']['column_index']
        sql_column_value = row['sql']['conds']['condition']
        new_and_old = construct_names(table_name, table_header, sql_column_index, sql_column_value)
        human_sql = table_end.sub(' ' + table_name, human_sql)
        human_sql = table_mid.sub(' ' + table_name + ' ', human_sql)
        for new_val, old_val in new_and_old:
            human_sql = human_sql.replace(old_val, new_val)
            question = ireplace(old_val, new_val, question)

        file.write(f"{question}\t{human_sql}\n")

with open('datasets/english2SQL-processed/eng2SQL_test.txt', 'w') as file:
    for row in tqdm(dataset['test']):
        tables.add(row['table']['name'])
        question = row['question']
        human_sql = row['sql']['human_readable']
        table_name = row['table']['name']
        if table_name == '':
            table_name = 'Unknown'
        table_header = row['table']['header']
        sql_column_index = row['sql']['conds']['column_index']
        sql_column_value = row['sql']['conds']['condition']
        new_and_old = construct_names(table_name, table_header, sql_column_index, sql_column_value)
        human_sql = table_end.sub(' ' + table_name, human_sql)
        human_sql = table_mid.sub(' ' + table_name + ' ', human_sql)
        for new_val, old_val in new_and_old:
            human_sql = human_sql.replace(old_val, new_val)
            question = ireplace(old_val, new_val, question)

        file.write(f"{question}\t{human_sql}\n")
    pass

# for row in tqdm(dataset['train']):
#     tables.add(row['table']['name'])
#     question = row['question']
#     human_sql = row['sql']['human_readable']
#     table_name = row['table']['name']
#     table_header = row['table']['header']
#     sql_column_index = row['sql']['conds']['column_index']
#     sql_column_value = row['sql']['conds']['condition']
#     new_and_old = construct_names(table_name, table_header, sql_column_index, sql_column_value)
#     human_sql = table_end.sub(' [' + table_name + ']', human_sql)
#     human_sql = table_mid.sub(' [' + table_name + '] ', human_sql)
#     for new_val, old_val in new_and_old:
#         human_sql = human_sql.replace(old_val, new_val)
#         question = question.replace(old_val, new_val)
#     print(row)
#     data = {
#          'question': row['question'],
#          'sql': row['sql']['human_readable'],
#          'table_header': row['table']['header'],
#          'table_header_types': row['table']['types'],
#          'conds': row['sql']['conds']['condition'],
#          'rows': row['table']['rows']
#     }

#     train_set.append(data)
#
# validation_set = []
# for row in tqdm(dataset['validation']):
#     tables.add(row['table']['name'])
#     data = {
#       'question': row['question'],
#       'sql': row['sql']['human_readable'],
#       'table_header': row['table']['header'],
#       'table_header_types': row['table']['types'],
#       'conds': row['sql']['conds']['condition'],
#       'rows': row['table']['rows']
#     }
#     validation_set.append(data)
#
# test_set = []
# for row in tqdm(dataset['test']):
#     tables.add(row['table']['name'])
#     data = {
#         'question': row['question'],
#         'sql': row['sql']['human_readable'],
#         'table_header': row['table']['header'],
#         'table_header_types': row['table']['types'],
#         'conds': row['sql']['conds']['condition'],
#         'rows': row['table']['rows']
#     }
#     test_set.append(data)
#
# train_df, validation_df, test_df = pd.DataFrame(train_set), pd.DataFrame(validation_set), pd.DataFrame(test_set)
#
# print(len(tables))
# print('OK')
