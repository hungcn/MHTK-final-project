# For re-populating the paths store in train.txt and valid.txt
# Run this file in folder darknet

import glob
import os

train_path = 'data/images/train'
valid_path = 'data/images/valid'

# Create train.txt and valid.txt
file_train = open('data/train.txt', 'w')
file_valid = open('data/valid.txt', 'w')

# Populate train.txt
for file in glob.iglob(os.path.join(train_path, '*.jpg')):
    file_train.write(file + "\n")

# Populate valid.txt
for file in glob.iglob(os.path.join(valid_path, '*.jpg')):
    file_valid.write(file + "\n")
