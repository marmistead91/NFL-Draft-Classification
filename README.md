# NFL-Draft-Classification
NFL Combine stats from 2000-2017 and classify whether a player would get drafted or not

# Data
Found data on Kaggle for stats from the 2000-2017 combine and draft results. For further exploration I got data for the 2018 and 2019 combine and draft to see how my model ran for those years.

# Cleaning and features
Dropped all columns that didn't have measurements, drills or draft info. Changed height to inches instead of ft-in and turned categorical data into dummy variables. 

# Model making
Turned 2000-2017 data and split into train-test-split to be able to train models and predict. Ran KNN, Logistic regression, random forest and XGBoost models. With all of these models I did GridSearch to tune all hyperparameters.

# Final model
XGBoost ended up being my best fitting model with an accuracy of 69% and F1 score of .79. These numbers are significant because 64% of players invited are drafted and I had a better accuracy for my predictions and a low amount of false positives. 

# Further Exploration
I obtained data for the 2018/2019 combine and draft and ran my models on them. For these years I needed a accuracy score higher than 60.2%. The data was cleaned so it can be ran in my model. When I ran the model I got an accuracy score of 63.7% and a F1 score of .76. My next step is to us the data from these years and add them to my original data to create a new model for the 2020 Draft.
