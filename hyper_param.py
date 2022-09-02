import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from telnetlib import GA
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from twilio.rest import Client

account_sid = 'AC2c24a6887a6eba1fa7a346380331e1e1'
auth_token = '2a77152f9c8cb363edc78b5bf8c5c601'
client = Client(account_sid, auth_token)

print("preparing the df...")
df = pd.read_excel('curr_alcs.xlsx')

print("casting the data...")
df['alc_percentage'] = df['alc_percentage'].astype(float)
df['if_sweet'] = df['if_sweet'].astype(bool)
df['rank'] = df['rank'].astype(int)
df['if_fruit_on_bottle'] = df['if_fruit_on_bottle'].astype(bool)
df['if_citrus'] = df['if_citrus'].astype(bool)
df['shooter_plastic'] = df['shooter_plastic'].astype(bool)
df = pd.get_dummies(data=df, columns=['type', 'brand', 'color'], dtype=bool)

print("dropping non-model related columns...")
df = df.drop(['name'], axis=1)

print("spliting into features...")
target = 'rank'
features = df.dtypes[(df.columns != target)].index 

print("split it fully...");
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

print("creating the model...")
model = XGBRegressor()

print("creating the params...")

# light collection of params:
params =  {
    'max_depth': [2, 3, 4],
    'n_estimators': [50, 100, 500, 1000],
    'colsample_bytree': [0.2, 0.3, 0.4],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.01, 0.05, 0.1],
    'subsample': [0.4, 0.6, 0.8]
}

print("creating the grid search...");
cv = GridSearchCV(estimator = model, param_grid = params, scoring = 'neg_mean_absolute_percentage_error', verbose = 2)

print("creating the fit...")
cv.fit(X_train, y_train)
print("- Best parameters:", cv.best_params_)
score_cv = abs(cv.best_score_)
print("- Best Score:", score_cv)
model_cv = cv.best_estimator_

# model.fit(X_train, y_train)
# print("- predicting...")
# y_pred = model.predict(X_test)

# score = mean_absolute_percentage_error(y_test, y_pred)
# print("Accuracy:", score)

print("predicting with the new model...")
y_pred = model_cv.predict(X_test)

score = mean_absolute_percentage_error(y_test, y_pred)
print("Accuracy:", score)

message = client.messages.create(
    body="done with rach job. \nbest cv score: {}\nbest predicted score: {}".format(score_cv, score),
    from_='+16203159839',
    to='+19702195822'
)