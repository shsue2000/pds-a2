import numpy as np
import pandas as pd 
import lightgbm as lgb 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv("train.tsv", sep="\t")
test_df = pd.read_csv("test.tsv", sep="\t")

X = train_df.drop(['#QueryID', 'Label','Docid'], axis=1)
y = train_df['Label']
testq = test_df['#QueryID']
testd = test_df['Docid']
test_X = test_df.drop(['#QueryID', 'Docid'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

def parameter_sweep():

    lgb_model = lgb.LGBMRegressor(
    task = 'predict',
    application = 'regression',
    objective = 'regression',
    n_estimators = 2500,
    learning_rate = 0.05,
    num_leaves=15,
    tree_learner='feature',
    max_depth =10,
    reg_sqrt='True',
    random_state=42,
    )    

    params = {
    'task': ['predict'],
    'n_estimators': [1000, 2500, 5000],
    'learning_rate': [0.05, 0.005],
    'num_leaves': [7, 31, 50],
    'max_depth': [10, 15],
    'early_stopping_rounds': [500],
    'min_data_in_leaf': [15, 25]
    }
        
    gs = GridSearchCV(lgb_model, params, scoring='r2', cv=5, verbose=1)    
    gs.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    score = gs.predict(test_X)
    
    df = pd.DataFrame({'QueryID': testq, 'DocID': testd, 'Score': score})    
    df.to_csv('A2.tsv', sep='\t', index=False, header=False)
    
if __name__=="__main__":
    sweep = False

if sweep == False:    

    gs = joblib.load("A2.pkl")
    score = gs.predict(test_X)

    df = pd.DataFrame({'QueryID': testq, 'DocID': testd, 'Score': score})
    df.to_csv('A2.tsv', sep='\t', index=False, header=False)
        
else:
    parameter_sweep()