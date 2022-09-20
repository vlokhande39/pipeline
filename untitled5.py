# -*- coding: utf-8 -*-
"""
pipeline - titanic dataset
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import set_config
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('C:\\Users\\SAI\\Desktop\\practice\\pipeline\\train.csv')
df_test = pd.read_csv("C:\\Users\\SAI\\Desktop\\practice\\pipeline\\test.csv")
df.head()
#drop some columns which is not required
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
# Step 1 -> train/test/split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),
                                                 df['Survived'],
                                                 test_size=0.2,
                                                random_state=42)
X_train.head()
y_train.sample(5)

# imputation transformer
trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')

# one hot encoding
trf2 = ColumnTransformer([
    ('ohe_sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')

# Scaling
trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])

# Feature selection
trf4 = SelectKBest(score_func=chi2,k=7)
# train the model
trf5 = DecisionTreeClassifier()

# Alternate Syntax
pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5)

# train
pipe.fit(X_train,y_train) #we already provide a ml algorithm in the pipeline
#so we are not providing the fit_transform function

# Code here
pipe.named_steps

# Display Pipeline

set_config(display='diagram')

# Predict
y_pred = pipe.predict(X_test)
y_pred

accuracy_score(y_test,y_pred)

# cross validation using cross_val_score

cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()

# gridsearchcv
params = {
    'trf5__max_depth':[1,2,3,4,5,None]
}

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

grid.best_score_
grid.best_params_

# export 
import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))
#export the picle file for production use

################################################################################
import pickle
import numpy as np
pipe = pickle.load(open('pipe.pkl','rb'))
# Assume user input
test_input2 = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'],dtype=object).reshape(1,7)
pipe.predict(test_input2)





