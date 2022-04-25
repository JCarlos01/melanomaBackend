# -*- coding: utf-8 -*-


#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import LeaveOneOut
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#Read dataset to pandas dataframe

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
#from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
import pickle



dataset2 = pd.read_csv('2Training80.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
dataset2t =pd.read_csv('2Testing20.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
# # data RGB
#dataset3 = pd.read_csv('2Training70(1).csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
#dataset3t = pd.read_csv('2Testing30(1).csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])

#dataset4 = pd.read_csv('7Training80(1).csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
#dataset4t =pd.read_csv('7Testing20(1).csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
# # data RGB
#dataset5 = pd.read_csv('7Training70(1).csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
#dataset5t = pd.read_csv('7Testing30(1).csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
                      
# X = dataset2.iloc[:,:-1]
# y = dataset2.iloc[:,-1]
# X_test = dataset2t.iloc[:,:-1]
# y_test = dataset2t.iloc[:,-1]
X2 = dataset2.iloc[:,:-1]
y2 = dataset2.iloc[:,-1]
X_test2 = dataset2t.iloc[:,:-1]
y_test2 = dataset2t.iloc[:,-1]
# #enfoque 2 SDA


X_train = X2
y_train = y2
X_test = X_test2
y_test = y_test2


 
#normalizacion
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_train = pd.DataFrame(X_train)

# scaler = MinMaxScaler()
# scaler.fit(X_test)
# X_test = scaler.transform(X_test)
# X_test = pd.DataFrame(X_test)
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train)
#scaler.fit(X_train)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)



with open('normalizacion.pkl','wb') as f:
    pickle.dump(scaler,f)

#scaler.fit(X_test)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# sampling_strategy={1:4000}#{1:4000,6:4000,7:1000,4:4000,5:2000,3:1000}
# ros = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=0)
# X_train, y_train = ros.fit_resample(X_train, y_train)

g = 1 #2**-6

c = 64 #2**14
# 
print('*****************************************************************')
print('best value of params')
print('c = ',c,'  g = ',g)
# 
print('*****************************************************************')
print('Prueba con el mejor parametro encontrado values')
modelo = (svm.SVC(kernel='rbf', gamma=g, C=c))
modelo.fit(X_train, y_train)
pred=modelo.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))

cm=confusion_matrix(y_test,pred)
res=matthews_corrcoef(y_test,pred)
print("mcc = ",res)
print("Balanced Accuracy Score (BAS):", balanced_accuracy_score(y_test, pred))
print("BER : ",((cm[0,1]/(cm[0,0]+cm[0,1]))+(cm[1,0]/(cm[1,0]+cm[1,1])))/2)
print("Recall Score :" , recall_score(y_test,pred, average ='weighted') )
print("Precision Score : ",precision_score(y_test,pred,average ='weighted'))
#res=matthews_corrcoef(y_test,pred)
#print("mcc = ",res)
print("f1-score",f1_score(y_test, pred, average='micro'))


# save
with open('svm.pkl','wb') as f:
    pickle.dump(modelo,f)

# 
# # print('\n')
# # print('error',c_model_error)
# # print('accuracy',c_model_acc)
# 
# # plt.figure(figsize=(6, 3))
# plt.plot(c, list_acc, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=2)
# plt.title('Accuracy SVM Value')
# plt.xlabel('c Values')
# plt.ylabel('Accuracys')
# 
# # plt.figure(figsize=(6, 3))
# # plt.plot(c, c_model_acc_error, color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=2)
# # plt.title('Error Rate SVM Value')
# # plt.xlabel('c Value')
# # plt.ylabel('Mean Error')
#    
# =============================================================================

