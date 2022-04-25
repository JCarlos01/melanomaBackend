
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import matthews_corrcoef
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import precision_score
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from numpy import absolute
from sklearn.metrics import accuracy_score, recall_score
#import matplotlib.pyplot as plt
#from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# import the regressor
#from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import pickle
# could use: import pickle... however let's do something else
#from sklearn.externals import joblib 

# this is more efficient than pickle for things like large numpy arrays
# ... which sklearn models often have.   

# then just 'dump' your file


# Read dataset to pandas dataframe
#Carac_HAM10000
#dataset = pd.read_csv('DataSet_Completo_2Clases.csv')
#,usecols=[1,2,3,5,6,7,8,9,11,12,13,15,16,17,18,21,23,24,25,26,27]


#Read dataset to pandas dataframe
#dataset = pd.read_csv('Caracteristicas.csv')
#train = pd.read_csv('2clases_DataAug_train.csv',usecols=[4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28])
train = pd.read_csv('2Training80.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]) # 0.74
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train)


with open('normalizacion.pkl','wb') as f:
    pickle.dump(scaler,f)

test = pd.read_csv('2Testing20.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]) 
#dataset = pd.read_csv('Caracteristicas.csv',usecols=[3,5,6,7,8,12,13,15,17,18,21,23,24,25,27]) 
#dataset = pd.read_csv('Caracteristicas.csv',usecols=[1,2,3,5,6,7,8,9,12,13,15,17,18,21,23,24,25,27])

X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]


X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)

#Implementing cross validation
p = 10 #numero de particiones

skf = StratifiedKFold(n_splits=p, random_state=None, shuffle=True)

partitions = skf.get_n_splits(X_train, y_train)
listarror = []
acc_error = []
acc_score = []
model_acc_error = []
model_acc_score = []

count_c = 0

exactitudMediaMaxima = 0

acc_error = []
acc_score = []
list_error = []
list_acc= []
list_params_n = []
list_params_r = []

for n in range(30, 31):
    
    model_acc_error = []
    model_acc_score = []
    
    for m in range(21 , 27, 2 ): #for para encontrar el mejor valor de gamma 
    
        # create regressor object
        clf = RandomForestClassifier(max_depth=m, random_state=0, n_estimators = n )
    
        cont = 0
        acc_error = []
        acc_score = [] 
        for train_index, test_index in skf.split(X_train, y_train):
            
            cont = cont + 1 #contador
            XP_train , XP_test = X_train.iloc[train_index,:],X_train.iloc[test_index,:]
            yp_train , yp_test = y_train[train_index] , y_train[test_index]
            
            # fit the regressor with x and y data
            clf.fit(XP_train, yp_train)
            
            pred_i=clf.predict(XP_test)

            acc = accuracy_score(pred_i , yp_test) #puntuacion de presicion 
            #acc = recall_score( pred_i , yp_test )
            
            #acc = matthews_corrcoef(pred_i , yp_test)
            
            err = (1.0 - acc)
            acc_score.append(acc)
            acc_error.append(err) #Error calculate
            cont = cont + 1 #contador
            
            #print( "n_estimators " , n , "max_depth", m,  "acc", acc)
            
        list_params_n.append(n)
        list_params_r.append(m)
        mean_acc_score = sum(acc_score)/p #media del scc
        mean_acc_error = sum(acc_error)/p #media del error almecenar que gamma y que c :
        list_error.append(mean_acc_error)
        list_acc.append(mean_acc_score)
        
        print(mean_acc_score)
        
acc_max = max(list_acc)
index = list_acc.index(acc_max)


print('*****************************************************************')
print('best value of params')
print(list_params_n[index],list_params_r[index],acc_max)
n_optimo=list_params_n[index]
val_r=list_params_r[index]


print('*****************************************************************')
print('Prueba con el mejor parametro encontrado values')

 # create regressor object
clf = RandomForestClassifier(max_depth=val_r, random_state=0, n_estimators = n_optimo )
  
# fit the regressor with x and y data
clf.fit(X_train, y_train)
pred=clf.predict(X_test)

# save
with open('model.pkl','wb') as f:
    pickle.dump(clf,f)

# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

pred2=clf2.predict(X_test[0:1])

print("max_depth", val_r, "n_estimators", n_optimo  )
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
print(matthews_corrcoef(y_test, pred))

print(classification_report(y_test[0:1], pred2))
print("\nresultado")
print(pred2)
# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title("Mean error n_estimators and max_depth values RandomForestClassifier",fontsize=14,fontweight="bold")
ax1.set_xlabel("n_estimators values")
ax1.set_ylabel("max_depth values")
ax1.set_zlabel("Mean error")
# Agregamos los puntos en el plano 3D
ax1.scatter(list_params_n, list_params_r, list_error, c='r', marker='o')
plt.show()

#Creamos la figura
fig1 = plt.figure()
# Creamos el plano 3D
ax2 = fig1.add_subplot(111, projection='3d')
ax2.set_title("Mean accuracy n_estimators and max_depth values RandomForestClassifier",fontsize=14,fontweight="bold")
ax2.set_xlabel("n_estimators values")
ax2.set_ylabel("max_depth values")
ax2.set_zlabel("Mean Accuracy")
# Agregamos los puntos en el plano 3D
ax2.scatter(list_params_n, list_params_r, list_acc, c='g', marker='o')
plt.show()
   




  