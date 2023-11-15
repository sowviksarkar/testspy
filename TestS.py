import os
import subprocess
import pandas as pd
import tensorflow as tf
import numpy as np
import random as r
from time import sleep

def get_functions_from_commit(commit_id):
    cmd_cscope = "git show {} -- *.c".format(commit_id)
    cmd_post = "grep @@ | awk -v n=5 '{ for (i=n; i<=NF; i++) printf \"%s%s\", $i, (i<NF ? OFS : ORS)}' | cut -d \"(\" -f 1 | grep -v -E 'DEFUN|ALIAS|struct|void|bool|EXIT_LABEL' | awk 'NF{ print $NF }' | sort -u | uniq | grep -v '{$' | grep -v '*$' | grep -v 'NBB$' | grep -v 'int$' | grep -v 'NBB_INT$' | grep -v 'NBB_VOID$'"
    command = cmd_cscope + ' | ' +cmd_post
    output = subprocess.check_output(command, shell=True).decode("utf-8")
    # lines = output.splitlines()[1:]  # Skip the header line
    functions = output.split()
    return(functions)

def get_locs(commit_id):
    cmd_glog = 'git show {} --pretty=tformat: --numstat'.format(commit_id)
    cmd_gawk = 'gawk \'{ add += $1; subs += $2; loc += $1 + $2 } END { printf "%s", loc }\' -'
    command = cmd_glog + ' | ' + cmd_gawk 
    output = subprocess.check_output(command, shell=True)
    return output

def get_model_fts(user_sha):
    final_functions=[]
    for func in user_commit_funcs:
        final_functions.append("func_{}".format(func))
    X_predict_df = pd.DataFrame()
    X_predict_df['functions_changed'] = final_functions

    from sklearn.preprocessing import MultiLabelBinarizer

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    one_hot_encoded_funcs = pd.DataFrame(mlb.fit_transform(X_predict_df['functions_changed']),
                                   columns=mlb.classes_,
                                   index=X_predict_df.index)

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    X_predict_df = pd.concat([X_predict_df, one_hot_encoded_funcs], axis=1)

    # Print the resulting DataFrame
    print(X_predict_df)

    Y_predict_df = model.predict_classes(X_predict_df)
    return(Y_predict_df)    


os.system('clear')
print("Welcome to tests.py - dynamic test case selection")
print("Please enter a halon commit id to continue: ")

user_sha = str(raw_input())

os.system('clear')
print('Commit ID entered: {}'.format(user_sha))
print('Attempting to mount halon repository...')
os.chdir('/ws/singhard/halon')
print('Success!')
raw_input("Press Enter to continue...")
os.system('clear')

print('Attempting to extract functions from commit...')
user_commit_funcs = get_functions_from_commit(user_sha)
for func in user_commit_funcs:
    print(func)
print("")
print('Success!')
print('Extracted {} functions'.format(len(user_commit_funcs)))
raw_input("Press Enter to continue...")
print("")

ft_list=set()
print('Finding statically mapped test scripts for functions...')
df = pd.read_csv('/ws/singhard/dbg/testspy/dataset/Static_Map.csv')
for func in user_commit_funcs:
    static_fts = df.loc[df['functions'] == func, 'test_cases']
    for ft in static_fts:
        ft1 = ft.split()
        for f in ft1:
            if f.startswith('['):
                f=f[1:]
            elif f.endswith(']'):
                f[:-1]
            f=f[1:-2]
            f=f+'.py'
            ft_list.add(f)
        # ft_list.append(ft.sp)
print('List of statically mapped FTs: ')
for ft in ft_list:
    print(ft)
print("")
print('Success!')
print('Total number of statically mapped FTs: {}'.format(len(ft_list)))
raw_input("Press Enter to continue...")
os.system('clear')

print('Loading saved binary prediction model...')
model = tf.keras.models.load_model('/ws/singhard/dbg/testspy/binary_predict_model.h5', compile=False)
model.summary()
print("")
print('Success!')
raw_input("Press Enter to continue...")
os.system('clear')

print("Predicting FTs using binary prediction model...")
sleep(4)
model_fts = get_model_fts(user_sha)
model_ft_list=set()

for model_ft in model_fts:
    model_ft_list.add(model_ft)

for model_ft in model_ft_list:
    print(model_ft)
    ft_list.add(model_ft)
print("")
print('Success!')
print('Total number of predicted FTs: {}'.format(len(model_ft_list)))
raw_input("Press Enter to continue...")
os.system('clear')

print("Overall set of FTs to run for the commit:")
for ft in ft_list:
    print(ft)
print("")
print('Success!')
print('Total number of FTs: {}'.format(len(ft_list)))
raw_input("Press Enter to continue...")
os.system('clear')

print('HT command: ')
ht_cmd = 'ht -t '
for ft in ft_list:
    ht_cmd = ht_cmd+ft+','
ht_cmd = ht_cmd[:-1]+'-h <platform> -i <image>'
print(ht_cmd)
print("")
print('Finished executing test spy!')
raw_input("Press Enter to exit...")
os.system('clear')