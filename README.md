# NFL-Draft-Classification
Every year roughly 330 players are invited to the NFL Scouting Combine to show off their athletic ability and skills to impress owners and coaches so they can have the opportunity to get drafted and play for an NFL team.

There are 6 main events that most players take part in
- 40 Yard Dash - measures speed, how fast can they run 40 yards
- 3 Cone Drill - measures agility and change of direction at high speed
- Bench Press - measures strength, counts number of repetitions of 225lbs
- Vertical Jump - measures explosiveness and how high they can jump
- Broad Jump - measures explosiveness, sees how far a player can jump horizontally 
- Shuttle Drill - measures lateral quickness and short distance explosiveness

Players don’t complete all of the drills, depending on their position. For example QB and Punters rarely compete in the bench press because you don’t have to be strong to kick or throw the ball accurately.

Using the NFL Combine stats from 2000-2017 I created a model that classifis whether a player would get drafted or not. 

# Data
The data is from Kaggle (https://www.kaggle.com/kbanta11/nfl-combine) for stats from the 2000-2017 combine and draft results. I also got the resutls for the 2018/2019 combine and draft to test on my model. For my models and graphs a player is either a 0 or a 1. 1 means that they were drafted and 0 means they were invited to the combine but were not drafted. 

# EDA
Average reps for bench press was higher for drafted players

![](NFL_class_pics/Bench_box.png)

Drafted players had a faster 3 cone drill, 40 yard dash, and shuttle box drill

![](NFL_class_pics/Cone_box.png) ![](NFL_class_pics/Forty_box.png) ![](NFL_class_pics/shuttle_box.png)

Draftedf players had bigger broad and vertical jumps

![](NFL_class_pics/broad_box.png) ![](NFL_class_pics/vert_box.png)

Most players that are invited to the Combine are drafted so I need to have an accuracy higher than 64% to consider my model successful 

![](NFL_class_pics/drafted_vs_non.png)

# Model making
Turned 2000-2017 data and split into train-test-split to be able to train models and predict. Ran KNN, Logistic regression, random forest and XGBoost models. With all of these models I did GridSearch to tuned all hyperparameters. The top 5 most important features in my model were the players bench repetitions, vertical jump, 3 cone drill time, weight, and their broad jump. 

![](NFL_class_pics/correlatiokn.png)

# Final model
XGBoost ended up being my best fitting model with an accuracy of 69% and F1 score of .79. These numbers are significant because 64% of players invited are drafted and I had a better accuracy for my predictions and a low amount of false positives. False positives means the model predicted that they were drafted when in reality they were not.

# Further Exploration
I obtained data for the 2018/2019 combine and draft and ran my models on them. For these years I needed a accuracy score higher than 60.2%. The data was cleaned so it can be ran in the same model. When I predicted the 2018/2019 class with my model I got an accuracy score of 63.7% and a F1 score of 0.76.
