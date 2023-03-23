## NOT USED ##

import csv as csv ## Used to Manage the Train and Test CSV files
import numpy as np

path_to_test_csv = '../data_files/df_prime_test.csv'
test_data_shape  = (7987, 12)
test_label_shape = (7987, 1)

path_to_train_csv = '../data_files/df_prime_train.csv'
train_data_shape = (24252, 12)
train_label_shape = (24252, 1)


#Features Choosen: [Eye_ID, Week_Num, Eye_Side, Frame_Num, Age, Gender, Race, Diabetes_Type, Diabetes_Years, BMI, BCVA, CST]

## Create the Training Label and Data arrays
train_data = np.zeros(train_data_shape)
train_label = np.zeros(train_label_shape)

with open(path_to_test_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    label_index = 0
    for row in reader:

        key_id = 0
        index  = 0
        for key in row.keys():
            if key_id in [3,4,5,6,8,9,10,11,12,13,14,16]:

                if key == 'Eye_side' :
                    if row[key] == 'OD':
                        train_data[label_index][index] = 0
                    if row[key] == 'OS':
                        train_data[label_index][index] = 1
                else:
                    train_data[label_index][index] = row[key]
                index  += 1
            if key_id in [15]:
                if row[key] == 35 or row[key] == 43:
                    train_label[label_index] = 0
                if row[key] == 47 or row[key] == 53:
                    train_label[label_index] = 1
                if row[key] == 61 or row[key] == 65 or row[key] == 71 or row[key] == 85:
                    train_label[label_index] = 2                               
            key_id += 1
        label_index += 1

## Create the Test Label and Data arrays
train_data = np.zeros(train_data_shape)
train_label = np.zeros(train_label_shape)

with open(path_to_test_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    label_index = 0
    for row in reader:

        key_id = 0
        index  = 0
        for key in row.keys():
            if key_id in [3,4,5,6,8,9,10,11,12,13,14,16]:

                if key == 'Eye_side' :
                    if row[key] == 'OD':
                        train_data[label_index][index] = 0
                    if row[key] == 'OS':
                        train_data[label_index][index] = 1
                else:
                    train_data[label_index][index] = row[key]
                index  += 1
            if key_id in [15]:
                if row[key] == 35 or row[key] == 43:
                    train_label[label_index] = 0
                if row[key] == 47 or row[key] == 53:
                    train_label[label_index] = 1
                if row[key] == 61 or row[key] == 65 or row[key] == 71 or row[key] == 85:
                    train_label[label_index] = 2                               
            key_id += 1
        label_index += 1




