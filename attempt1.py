from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays

import problem

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

fr_holidays = holidays.FR()

def time_of_day(hour):
    if hour > 3 and hour <= 6:
        return 0
    elif hour > 6 and hour <= 10:
        return 1
    elif hour > 10 and hour <= 13:
        return 2
    elif hour > 13 and hour <= 17:
        return 3
    elif hour > 17 and hour <= 22:
        return 4
    return 5

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    # X.loc[:, "bank_holiday"] = X["date"].apply(lambda x: 1 if x in fr_holidays else 0)
    # X.loc[:, "week-end"] = X["weekday"].apply(lambda x: 1 if x in [5, 6] else 0)
    # X.loc[:, "time_of_day"] = X["hour"].apply(lambda x: time_of_day(x))

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()
X_final = problem.get_final_test_data()

def _encode_categorical(X):
    X = X.copy()

    return X.drop(columns=['site_name', 'counter_technical_id', 'counter_name','latitude', 'longitude'])

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = FunctionTransformer(_encode_categorical)
    categorical_cols = ["counter_id", "site_id", "counter_installation_date"]

    preprocessor = ColumnTransformer(
        [
            ("date", StandardScaler(), date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    regressor = XGBRegressor(n_estimators=1000, max_depth=25, eta=0.1, subsample=0.9, colsample_bytree=0.6)
    # regressor = GradientBoostingRegressor(n_estimators=30, max_depth=25)
    # regressor = Ridge()
    # regressor = LinearRegression()

    pipe = make_pipeline(date_encoder, categorical_encoder, preprocessor, regressor)

    return pipe

pipe = get_estimator()
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_final)

print(f"Test set, RMSE={mean_squared_error(y_test, pipe.predict(X_test), squared=False):.2f}")

#cv = TimeSeriesSplit(n_splits=6)

#scores = cross_val_score(
#    pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
#)
#print("RMSE: ", scores)
#print(f"RMSE (all folds): {-scores.mean():.3} ± {(-scores).std():.3}")

results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)

results.to_csv("submission.csv", index=False)
