import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_and_concat(file):
    # combine all csvs into one df
    finaldf = pd.read_csv(file[0])
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
    # drop draft round
    data = data.drop(columns=['DraftRd'])
    return data

def dummy(data, dum):
#   create dummies for positions
    dummies = pd.get_dummies(data[dum], prefix= dum)
    data = data.drop([dum], axis=1)
    data = pd.concat([data, dummies], axis=1)
    return data

def testing_data(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
    return X_train, X_test, y_train, y_test

def rearrange_cols(data):
    data = data[['Height', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'Broad_Jump', 'Cone',
       'Shuttle', 'Drafted', 'Pos_CB', 'Pos_DE', 'Pos_DT', 'Pos_FB', 'Pos_ILB',
       'Pos_LS', 'Pos_OG', 'Pos_OLB', 'Pos_OT', 'Pos_P', 'Pos_RB', 'Pos_S',
       'Pos_TE', 'Pos_WR', 'Pos_EDGE', 'Pos_LB', 'Pos_DL', 'Pos_OL', 'Pos_C']]
    return data

# def fit_pred(model, test= acc, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     if test == 'acc':
#         test_score = accuracy_score(y_test, preds)
#         print('Accuracy :')
#     else:
#         test_score = f1_score(y_test, preds)
#         print('F1:')
#     return test_score