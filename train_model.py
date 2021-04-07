import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

WineData=pd.read_csv('QualityPrediction.csv')

#Removing Duplicates
WineData.drop_duplicates(keep='first',inplace=True)

#defining a function to identify the upper and lower range for outliers i.e +- 1.5 IQR 
def OutlierRange(Feature):
    Q1=np.percentile(Feature,[25], interpolation='midpoint',)
    Q3=np.percentile(Feature,[75],interpolation='midpoint')
    IQR=Q3-Q1
    IQR1_5 = 1.5 * IQR
    Upper_Range = Q3+IQR1_5
    Lower_Range = Q1 - IQR1_5
    return Upper_Range, Lower_Range

# defining a function to drop the outliers from the respected fields
def DropOutlier(Feature,Upper_Range,Lower_Range):
    WineData.drop(WineData[Feature> Upper_Range[0]].index,axis=0,inplace=True)
    WineData.drop(WineData[Feature<Lower_Range[0]].index,axis=0,inplace=True)

# removing outlier from each column
Feature=WineData['fixed acidity']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

Feature=WineData['volatile acidity']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

Feature=WineData['citric acid']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

# Feature=WineData['residual sugar']
# Upper_Range,Lower_Range=OutlierRange(Feature)
# DropOutlier(Feature,Upper_Range,Lower_Range)

Feature=WineData['chlorides']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

Feature=WineData['free sulfur dioxide']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

Feature=WineData['total sulfur dioxide']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

# Feature=WineData['density']
# Upper_Range,Lower_Range=OutlierRange(Feature)
# DropOutlier(Feature,Upper_Range,Lower_Range)

# Feature=WineData['pH']
# Upper_Range,Lower_Range=OutlierRange(Feature)
# DropOutlier(Feature,Upper_Range,Lower_Range)

Feature=WineData['sulphates']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

Feature=WineData['alcohol']
Upper_Range,Lower_Range=OutlierRange(Feature)
DropOutlier(Feature,Upper_Range,Lower_Range)

# Splitting Data into X and Y Variables
X=WineData.drop(['quality','pH','residual sugar','density'],axis=1)
Y=WineData['quality']

#Balancing the Class by Oversampling
oversample = SMOTE(k_neighbors=1)
X, Y = oversample.fit_resample(X, Y)
#Traing Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

#KNN

# Feature Scaling
StandardScaler=StandardScaler()
X_train_fs=StandardScaler.fit_transform(X_train)
X_train.to_csv('X_train.csv',index=False)
X_test_fs=StandardScaler.transform(X_test)

# Fit the model on training set
model=KNeighborsClassifier(n_neighbors=4)
model.fit(X_train_fs,Y_train)

#accuracy
accuracy=model.score(X_train_fs,Y_train)
print(accuracy)

# #Random Forest
# # Fit the model on training set
# model=RandomForestClassifier(random_state=5,n_estimators=100,max_leaf_nodes=50,max_depth=7,bootstrap=False,criterion='gini',min_samples_split=2,min_samples_leaf=1)
# model.fit(X_train,Y_train)

# #accuracy
# accuracy=model.score(X_train,Y_train)
# print(accuracy)

#save the model to disk
filename='finalized_model.pkl'
joblib.dump(model,filename)

model=joblib.load('finalized_model.pkl')

