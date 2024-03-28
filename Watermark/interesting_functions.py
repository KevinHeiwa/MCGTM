import re

case_1 = ["def","class","print","pprint","for","while"]  #"int","float","str"
case_2 = ["=","==","#",">","<",">=","<="]  #,"\t#","//","!="
case_3 = ["(","[","'",'"','{']
case_4 = ['"""',"'''","```"]

def find_first_match(string):
    if '"""' in string:
        return string
    else:
        for char in string:
            if char in case_2 or char in case_3:
                return char

    return None 

def include_case(token):
    for case in case_4:
        if case in token:
            return True
    for case in case_3:
        if case in token:
            return True
    for case in case_2:
        if case in token:
            return True
    for case in case_1:
        if case.strip() == token:
            return True
    return False

def include_space(token):
    if re.match(r'^\s*$', token):
        return True
    return False

def has_whitespace(token):
    return ' ' in token

def is_whitespace(token):
    if token.strip() == '':
        return True
    return False

def check_teet_1_in_case_pure(token):
    pattern_1 = r'^def$'
    match_1 = re.match(pattern_1, token)
    pattern_2 = r'^class$'
    match_2 = re.match(pattern_2, token)
    pattern_3 = r'^print$'
    match_3 = re.match(pattern_3, token)
    pattern_4 = r'^pprint$'
    match_4 = re.match(pattern_4, token)
    pattern_5 = r'^for$'
    match_5 = re.match(pattern_5, token)
    pattern_6 = r'^while$'
    match_6 = re.match(pattern_6, token)
    if match_1 or match_2 or match_3 or match_4 or match_5 or match_6:
        return True
    else:
        return False
    

def check_already(token):
    if '(' in token and ')' in token:
        return True
    if '[' in token and ']' in token:
        return True
    if '{' in token and '}' in token:
        return True
    if "'" in token:
        count = token.count("'")
        if count == 2:
            return True
        else:
            return False
    if '"' in token:
        count = token.count('"')
        if count == 2:
            return True
        else:
            return False
    return False


def count_true_elements(lst):
    count = 0
    for item in lst:
        if item == 'True':
            count += 1
    return count

def count_true_elements(lst):
    count = 0
    for item in lst:
        if item == 'True':
            count += 1
    return count

def count_brackets(string):
    count_parentheses = 0
    count_brackets = 0
    count_braces = 0
    for char in string:
        if char == '(':
            count_parentheses += 1
        elif char == '[':
            count_brackets += 1
        elif char == '{':
            count_braces += 1           
    return count_parentheses, count_brackets, count_braces


def count_brackets_Mirror(string):
    count_parentheses = 0
    count_brackets = 0
    count_braces = 0
    
    for char in string:
        if char == ')':
            count_parentheses += 1
        elif char == ']':
            count_brackets += 1
        elif char == '}':
            count_braces += 1          
    return count_parentheses, count_brackets, count_braces

# def remove_elements(dictionary, count):
#     total_removed = 0
#     keys = sorted(dictionary.keys(), reverse=True)
#     for key in keys:
#         sub_dict = dictionary[key]
#         sub_keys = sorted(sub_dict.keys(), reverse=True)
#         for sub_key in sub_keys:
#             del sub_dict[sub_key]
#             total_removed += 1
#             if total_removed == count:
#                 return

def remove_elements(dictionary, count):
    total_removed = 0
    keys = sorted(dictionary.keys(), reverse=True)
    for key in keys:
        del dictionary[key]
        total_removed += 1
        if total_removed == count:
            return

def remove_elements_from_end(lst, count):
    count = min(count, len(lst))
    for i in range(count):
        lst.pop()

def replace_elements_from_end(lst, count):
    count = min(count, len(lst))           
    for i in range(count):
        lst[-(i + 1)] = "False"

