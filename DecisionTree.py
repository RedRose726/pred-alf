import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

balance_data=pd.read_csv('C:/Users/LENOVA/PycharmProjects/mainpro/static/indian_liver_patient(origin).csv',sep=',')
#print(balance_data.head())
#balance_data.info()   #to locate if there are any null values
#print(balance_data['Dataset'].value_counts()) #to display no.of LF and no LF
        #to set the values as 1 n 0
balance_data['Dataset']=balance_data['Dataset'].map({2:0,1:1})
#print(balance_data['Dataset'])
        #to display missing value rows
#missrows=balance_data[balance_data.isnull().any(axis=1)]
#print(missrows)
        #to fill it by mean of the corresponding values
#X=balance_data.iloc[:,:-1].values    #seperating result with other datas
#Y=balance_data.iloc[:,-1].values
#imp=Imputer(missvalues='NaN', strategy='mean',axis=0)
#X[:,9:10]=imp.fit_transform(X[:,9:10])
        #to fill missing values
balance_data['Albumin_and_Globulin_Ratio'].fillna(value=0,inplace=True)
#balance_data.info()
        #to plot histogram
#cntcls=pd.value_counts(balance_data['Dataset'],sort=True).sort_index()
#cntcls.plot(kind='bar')
#plt.title("histogram")
#plt.xlabel("Dataset")
#plt.ylabel("Frequency")

X=balance_data.iloc[:,:-1].values    #seperating result from other datas
Y=balance_data.iloc[:,-1].values
#X = balance_data.values[1:250, 0:9]            #seperating result from other datas
#Y = balance_data.values[1:250, 10]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)       #training n testing data
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

        #to check whether to scale or not
#print(balance_data.describe())
        #scaling
sc=StandardScaler()
#X_train[:,2:]=sc.fit_transform(X_train[:,2:])
#X_test[:,2:]=sc.transform(X_test[:,2:])

test_data=[[17,1,0.7,0.2,145,18,36,7.2,3.9,1.18]]
test=[[17,0,0.9,0.2,279,40,46,7.3,4,1.2]]
test1=[[17,0,0.9,0.3,202,22,19,7.4,4.1,1.2]]
test4=[[60,0,0.5,0.1,500,20,34,5.9,1.6,0.37]]

test2=[[17,0,0.9,0.2,224,36,45,6.9,4.2,1.55]]    #diseased(in dataset but not in ytest)got correct when scaled   wrong
test3=[[36,0,0.8,0.2,158,29,39,6,2.2,0.5]]       #not diseased(in dataset but not in ytest)didnt get correct when scaled  wrong
test5=[[32,0,12.1,6,515,48,92,6.6,2.4,0.5]]      #diseased(correct in both)correct in both



                     #Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="gini",random_state=0,min_samples_leaf=5)
dt.fit(X_train,Y_train)
print("xtest dt= ",X_test)
dt_predict = dt.predict(X_test)
print("prediction using DT:",dt_predict)

accuracy2=accuracy_score(Y_test,dt_predict)
print("The accuracy of DT algorithm is: ",str(accuracy2*100),"%")

dt_predict1=dt.predict(test2)
print("dt1=",dt_predict1)
probs=dt.predict_proba(test2)
print("probs=",probs)

cm= confusion_matrix(Y_test,dt_predict)
print("CM : ",cm)

rocauc=roc_auc_score(Y_test,dt_predict)
print("The roc of DT algorithm is: ",rocauc)

#joblib.dump(dt,'dt2.pkl')


