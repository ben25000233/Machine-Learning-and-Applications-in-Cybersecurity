import os
import pandas as pd
import shutil
df = pd.read_csv(
    './dataset.csv', low_memory=False)
df = df[['filename', 'label']]
labelcsv = df[(df['label'] == 'Mirai') | (
    df['label'] == 'Bashlite') | (df['label'] == 'Unknown')]
path = './img_data/all/'
train_folder = "./img_data/train/"
test_folder = "./img_data/test/"
if not os.path.isdir(train_folder):
    os.makedirs(train_folder)
if not os.path.isdir(test_folder):
    os.makedirs(test_folder)
files = os.listdir(path)
folders = os.listdir(path)
catagory = []
shot = 2
for file in files:
    src = path+file
    train_dst = train_folder+file
    test_dst = test_folder+file
    filename = file.replace('.jpg', '')
    match = labelcsv[labelcsv['filename'] == filename]
    # if catagory.count(match.iat[0, 1]) < shot and not os.path.isfile(dst):
    if(not match.empty):
        if catagory.count(match.iat[0, 1]) < shot:
            catagory.append(match.iat[0, 1])
            shutil.copy(src, train_dst)
        else:
            shutil.copy(src, test_dst)
print(catagory)
