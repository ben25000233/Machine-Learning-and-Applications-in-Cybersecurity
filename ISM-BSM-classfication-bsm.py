#!/usr/bin/env python
# coding: utf-8

# In[1]:


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


# In[2]:


import os
import pandas as pd
import numpy as np
from pathlib import Path
#from sklearn import preprocessin
path = "C:/Users/ben/Desktop/malware-1"
filess = os.listdir(path)
malware = []
mal_name = []
t_mal = []
t_lable = []
for i in filess :
    mono_path = path + "/" + i
    if os.path.isdir(path): #判斷是資料夾還是文件
        if not os.listdir(path): # 如果資料夾為空
            continue
        else:
            files = os.listdir(mono_path)
            for j in files :
                if len(malware) < 10000 and os.path.getsize(mono_path+"/"+j) > 0:
                    mono_name = Path(mono_path+"/"+j).stem
                    temp_name = mono_name.split('-')
                    mono_name = temp_name[0]
                    mal_name.append(mono_name)
                    data = pd.read_csv(mono_path+"/"+j, low_memory = False)
                    pid = data[["PID"]]
                    pid = np.array(pid)
                    std_pid = pid[0]
                    pid -= std_pid
                    sys = data[["SYSCALL"]]
                    sys = np.array(sys)
                    mono_data = []
                    if len(sys) > 5000:
                        for k in range(5000):
                            temp = []
                            temp.append(pid[k][0])
                            temp.append(sys[k][0])
                            mono_data.append(temp)
                    else:
                        for k in range(len(sys)):
                            temp = []
                            temp.append(pid[k][0])
                            temp.append(sys[k][0])
                            mono_data.append(temp)
                    malware.append(mono_data)
                else:
                    break
print(len(malware))  


# In[3]:


temp_sys_cat = []
for i in range(len(malware)):
    for j in range(len(malware[i])):
        temp_sys_cat.append(malware[i][j][1])
print(temp_sys_cat[0:50])


# In[4]:


sys_cat = set()
for i in range(len(temp_sys_cat)):
    sys_cat.add(temp_sys_cat[i])
sys_cat = list(sys_cat)
print(sys_cat)


# In[5]:


from sklearn import preprocessing
std_sys = []
for i in range(len(sys_cat)):
    std_sys.append(i)
std_sys = np.reshape(std_sys, (-1, 1))
zscore = preprocessing.StandardScaler()
std_sys = zscore.fit_transform(std_sys)


# In[6]:


sys_dic = {}
for i in range(len(sys_cat)):
    sys_dic[sys_cat[i]] = std_sys[i]


# In[7]:


for i in range(len(malware)):
    for j in range(len(malware[i])):
        if is_number(malware[i][j][1]) == False:
            malware[i][j][1] = sys_dic[malware[i][j][1]]
            malware[i][j][1] = malware[i][j][1][0]
            malware[i][j] = list(malware[i][j])


# In[8]:


t_mal = []
t_name = []
temp = []
for i in range(len(malware)):
    temp = list(malware[i])
    if len(temp)>1000:
        t_mal.append(temp[0:1000])
        t_name.append(mal_name[i])
    elif len(temp) > 50:
        a = 1000 - len(temp)
        A = [[0]*2 for _ in range(a)]
        temp += A
        t_mal.append(temp)
        t_name.append(mal_name[i])
print(len(t_mal), len(t_name))


# In[10]:


path = "C:/Users/ben/Desktop/dataset.csv"
read_lable = pd.read_csv(path)
read_lable = read_lable[["filename","label"]]
temp_lable = read_lable.to_numpy()
print(temp_lable[0])


# In[11]:


lable_dic = {}
for i in range(len(temp_lable)):
    lable_dic[temp_lable[i][0]] = temp_lable[i][1]


# In[12]:


lable = []
for i in range(len(t_name)):
    lable.append(lable_dic[t_name[i]])
print(lable[0:50])


# In[13]:


req = set()
for i in range(len(lable)):
    req.add(lable[i])
req = list(req)
print(req)


# In[14]:


for i in range(len(lable)):
    for j in range(len(req)):
        if lable[i] == req[j]:
            lable[i] = j
            break
lables = np.eye(len(req))[lable]
print(lable[0:50])


# In[15]:


hsum = [0]*8
for i in range(len(lable)):
    hsum[lable[i]]+= 1
print(hsum)


# In[16]:


from tkinter import _flatten
temp = []
for i in range(len(t_mal)):
    temp.append(list(_flatten(t_mal[i])))


# In[37]:


bsm = []
for i in range(len(temp)):
    mono = []
    temp_id = 0
    temp_name = 0
    count = 0
    for j in range(0, len(temp[i]),2):
        if count == 5:
            mono.append(temp_id)
            mono.append(temp_name)
            temp_id = temp[i][j]
            temp_name = temp[i][j+1]
            count = 1
        else:
            temp_id += temp[i][j]
            temp_name += temp[i][j+1]
            count += 1
        if j == len(temp[i]) - 2:
            mono.append(temp_id)
            mono.append(temp_name)
    bsm.append(mono)
print(len(bsm[0]), len(temp[0]))


# In[40]:


print(len(bsm), len(temp))


# In[42]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import KFold
num_folds = 10
acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits=5, shuffle=True)
fold_no = 1
np_temp = np.array(temp)

for i in range(len(lable)):
    for j in range(len(req)):
        if lable[i] == req[j]:
            lable[i] = j
            break
lables = np.eye(len(req))[lable]
np_lable = np.array(lables)

for train, test in kfold.split(bsm, lable):
    model = Sequential()
    model.add(Dense(1024, input_dim=2000))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('--------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    train_history = model.fit(np_temp[train],  
                              np_lable[train],  
                              epochs=10, batch_size=300)
    
    scores = model.evaluate(np_temp[test], np_lable[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
    del model

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for ten folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation = train_test_split(temp,lable, test_size = 0.3,shuffle = True)
from sklearn.model_selection import cross_val_score

#using DecisionTreeClassifier to train
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
scores = cross_val_score(model,temp,lable,cv=2,scoring='accuracy')
print("DecisionTree: ", scores.mean())


#using RandomForestClassifier to train
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
scores = cross_val_score(model,temp,lable,cv=2,scoring='accuracy')
print("RandomForestClassifier: ", scores.mean())


#using KNeighborsClassifier to train
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
scores = cross_val_score(model,temp,lable,cv=2,scoring='accuracy')
print("KNeighborsClassifier: ", scores.mean())


# In[19]:


'''


import matplotlib.pyplot as plt
plt.plot(train_history.history["accuracy"])
plt.plot(train_history.history["val_accuracy"])
plt.title("Train History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"], loc="lower right")
plt.show()
plt.plot(train_history.history["loss"])
plt.plot(train_history.history["val_loss"])
plt.title("Train History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "validation"], loc="upper left")
plt.show()
'''


# In[ ]:




