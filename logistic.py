import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

'''

features(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'WikiId', 'Name_wiki',
       'Age_wiki', 'Hometown', 'Boarded', 'Destination', 'Lifeboat', 'Body',
       'Class'],
      dtype='object')
'''
file_path='C:/Users/DELL/Desktop/dataets/train.csv'
train_data=pd.read_csv(file_path)  

#print(train_data)



#print(train_data.columns)

## dreoping the usless features from train data_sets;
train_data.drop(['Age_wiki', 'Hometown','Destination','Name_wiki','WikiId','Cabin','Age_wiki','Ticket','Name','Fare','PassengerId'],inplace=True,axis=1)

train_data.drop(['Lifeboat','Boarded','SibSp'],inplace=True,axis=True)
train_data.drop(['Body'],inplace=True,axis=1)
#print(train_data.shape)
#print(train_data.head(10))



avrage_Age=np.mean(train_data["Age"])

train_data["Age"]=train_data["Age"].fillna(np.absolute(avrage_Age))

train_data.dropna(inplace=True)

print(train_data.isnull().sum())

#print(X_train.shape," ",y_train.shape)

## changing the datatype 
train_data["Age"]=train_data["Age"].astype("int64")
train_data["Survived"]=train_data["Survived"].astype("int64")
train_data["Class"]=train_data["Class"].astype("int64")


## convert clasiification into 0 OR 1;
#print(train_data.head())

embark = pd.get_dummies(train_data["Embarked"],drop_first=True)

sex=pd.get_dummies(train_data["Sex"],drop_first=True)

pclass=pd.get_dummies(train_data["Pclass"],drop_first=True)




train_data=pd.concat([train_data,sex,embark,pclass],axis=1)

#train_data.drop(["Sex",'Embarked','Pclass','Class'],axis=1,inplace=True)


X_train=train_data.iloc[:,1:]
y_train=train_data.iloc[:,0]
#print(X_train)
'''
## define the sigmoid function;
def sigmoid(score):
       return 1/(1+np.exp(-score))


def log_likehood(feature,target,weights):
       score=np.dot(feature,weights)
       ll=np.sum(target*score - np.log(1+np.exp(score)))

       return ll 


## building the main logic

def logistic_Regression(feature,target,num_steps,learning_rate,add_intercept=False):
       if add_intercept:
              intercept=np.ones((feature.shape[0],1))
              feature=np.hstack((intercept,feature))

       weights = np.zeros(feature.shape[1])

       for step in range(num_steps):

              score = np.dot(feature,weights)

              prediction = sigmoid(score)
              
              outpu_erroe_signal=target-prediction

              gradient = np.dot(feature.T,outpu_erroe_signal)

              weights += learning_rate*gradient

              if step % 1000 == 0:

                     print(log_likehood(feature,target,weights))

       return weights           


weights = logistic_Regression(X_train, y_train,
                     num_steps = 1000, learning_rate = 0.000009, add_intercept=True)

print(weights)


## accuracy:

##data_with_intercept=np.hstack((np.ones((X_train.shape[0],1)),y_train))
##simulated_labels).sum().astype(float) / len(preds)))final_score=np.dot(data_with_intercept,weights)
'''
'''
preds=np.rounp.concatenatend(sigmoid(final_score))

print("Accuracy from scratch:{0}".formate((preds == simulated_labels).sum().astype(float) / len(preds)))

'''
'''
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 1e15)
clf.fit(X_train, y_train)

print(clf.score(X_train,y_train))




final_scores = np.dot(np.hstack((np.ones((X_train.shape[0], 1)),X_train)), weights)
preds = np.round(sigmoid(final_scores))

print ('Accuracy from scratch: {0}'.format((preds==y_train).sum().astype(float) / len(preds)))

'''