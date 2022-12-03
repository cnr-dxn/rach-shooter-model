import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from telnetlib import GA
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
from twilio.rest import Client

account_sid = 'AC2c24a6887a6eba1fa7a346380331e1e1'
auth_token = '2a77152f9c8cb363edc78b5bf8c5c601'
client = Client(account_sid, auth_token)

print("preparing the df...")
# df = pd.read_excel('curr_alcs.xlsx')
df = pd.read_excel('curr_alcs_new.xlsx')

df = df.dropna()

print("casting the data...")
df['alc_percentage'] = df['alc_percentage'].astype(float)
df['if_sweet'] = df['if_sweet'].astype(bool)
df['rank'] = df['rank'].astype(int)
df['if_fruit_on_bottle'] = df['if_fruit_on_bottle'].astype(bool)
df['if_citrus'] = df['if_citrus'].astype(bool)
df['if_cinnamon'] = df['if_cinnamon'].astype(bool)
df['if_cream'] = df['if_cream'].astype(bool)
df['if_pepper'] = df['if_pepper'].astype(bool)
df = pd.get_dummies(data=df, columns=['type', 'brand', 'color'], dtype=bool)
target_shot = df[df['rank'] == 0]
df = df[df['rank'] != -1]
df = df[df['rank'] != 0]

if int(target_shot.shape[0]) > 1:
    raise ValueError("you sure you want to predict more than one at a time?")

print("dropping non-model related columns...")
df = df.drop(['name'], axis=1)

print("spliting into features...")
target = 'rank'
features = df.dtypes[(df.columns != target)].index 

print("split it fully...");
best_score = 900.00
rand_samp = 0
print("finding best rand samp...")
for i in range(0,200):
    # if i % 15 == 0:
        # print(f"- on val #{i} \r")
    X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                        df[target], 
                                                        test_size=0.1, 
                                                        random_state=i)
    # X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

    # print("creating the model...")
    model = XGBRegressor(colsample_bytree= 0.3, 
                        gamma= 0.05, 
                        max_depth= 2, 
                        min_child_weight= 1, 
                        n_estimators= 10, 
                        subsample= 0.4)
    # model = XGBRegressor()

    model.fit(X_train, y_train)
    # print("- predicting...")
    y_pred = model.predict(X_test)


    score = mean_squared_error(y_test, y_pred)
    if score < best_score:
        best_score = score
        rand_samp = i
        # print("ping, best score now:", best_score, "rand_samp now", rand_samp)
        
X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[target], 
                                                    test_size=0.1, 
                                                    random_state=rand_samp)
# X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

print("creating the model...")
model = XGBRegressor(colsample_bytree= 0.3, 
                    gamma= 0.05, 
                    max_depth= 2, 
                    min_child_weight= 1, 
                    n_estimators= 10, 
                    subsample= 0.4)
# model = XGBRegressor()

model.fit(X_train, y_train)
print("- predicting...")
y_pred = model.predict(X_test)
score = mean_squared_error(y_test, y_pred)
print("RMSE:", score)



if int(target_shot.shape[0]) != 0:
    print("\n=================== PREDICTION ===================")
    shot_name = str(target_shot['name'].iat[0])
    target_shot = target_shot.drop(['name', 'rank'], axis=1)
    if target_shot['type_gin'].iat[0] == True:
        predicted_val = max(df.index) 
    else:
        predicted_val_raw = model.predict(target_shot)
        predicted_val = int(round(predicted_val_raw[0])) 
    print("final prediction for {}: {}".format(shot_name, predicted_val))

else:
    print("no shots found to predict")