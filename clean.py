import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

# %matplotlib inline

def read_and_concat(file):
    # combine all csvs into one df
    finaldf = pd.read_csv('nfl-combine/2000Offense.csv')
    for i in file:
        df = pd.read_csv(i)
        finaldf = finaldf.append(df)
        finaldf.reset_index(drop=True, inplace = True)
    return finaldf

def convert_height(x):
    # convert ft'in'' to in
    feet = x.split("-")[0]
    inches = x.split("-")[1]
    height = (int(feet) * 12) + int(inches)
    return height

def clean_data(data):
# drop unnecessary columns
    data.drop(columns=['Rk', 'College', 'Player', 'AV', 'School', 'Year'], inplace=True)
# change names so they will fun in ols
    data = data.rename(columns={'Broad Jump': 'Broad_Jump', '40YD': 'Forty', '3Cone': 'Cone'})
    data['Height'] = data['Height'].apply(convert_height)
# get draft year and round
    data["Drafted (tm/rnd/yr)"] = data["Drafted (tm/rnd/yr)"].where(pd.notnull(data["Drafted (tm/rnd/yr)"]), None)
    data["DraftRd"] = [x.split(" / ")[1] if x != None else None for x in data["Drafted (tm/rnd/yr)"]]
    data["DraftRd"] = data["DraftRd"].str.replace('[a-zA-Z]+', '')
    data = data.drop(["Drafted (tm/rnd/yr)"], axis=1)
# fill na to 8th round which means they were not drafted
    data['DraftRd'] = data['DraftRd'].fillna(8)
    data = data.apply(pd.to_numeric, errors='ignore')
# create new column 0=undrafted 1=drafted
    data['Drafted'] = data['DraftRd'].apply(lambda x: 0 if x == 8 else 1)
#     clear na so models can run
    data = data.fillna(0)
#   create dummies for positions
    pos_dummies = pd.get_dummies(data['Pos'], prefix='pos', drop_first=True)
    data = data.drop(['Pos'], axis=1)
    data = pd.concat([data, pos_dummies], axis=1)
    # drop draft round
    data = data.drop(columns=['DraftRd'])
    return data

def testing_data(data, target):
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
    return X_train, X_test, y_train, y_test