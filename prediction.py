from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor  
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pandas as pd 

df = pd.read_csv("outputData.csv")

label_encoder = LabelEncoder()

df['city_code'] = label_encoder.fit_transform(df['city'])

df.drop('city', axis=1, inplace=True)


X = df.drop(['price', 'date','street','statezip','country'], axis=1)
Y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=46)

#Randomforest
model_rf = RandomForestRegressor()
model_rf.fit(X_train,y_train)
pred_y_rf = model_rf.predict(X_test)
print("Randomforest")
print("ΜΑΕ:",mean_absolute_percentage_error(y_test, pred_y_rf))
r2_rf = r2_score(y_test,pred_y_rf)
print("real r2:",r2_rf)

#LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
pred_y_lr = model_lr.predict(X_test)
print("LinearRegression:")
print("MAE:",mean_absolute_percentage_error(y_test, pred_y_lr))
r2_lr = r2_score(y_test,pred_y_lr)
print("real r2:",r2_rf)

#XGBoost
model_XGB = XGBRegressor()
model_XGB.fit(X_train,y_train)
pred_y_xgb = model_XGB.predict(X_test)
print("XGBoost:")
print("MAE:",mean_absolute_percentage_error(y_test, pred_y_xgb))
r2_xg = r2_score(y_test,pred_y_xgb)
print("real r2:",r2_rf)

#CatBoost
model_cb = CatBoostRegressor()
model_cb.fit(X_train,y_train)
pred_y_cb = model_cb.predict(X_test)
print("CatBoost:")
print("MAE:",mean_absolute_percentage_error(y_test, pred_y_cb))
r2_cb = r2_score(y_test,pred_y_cb)
print("real r2:",r2_cb)






