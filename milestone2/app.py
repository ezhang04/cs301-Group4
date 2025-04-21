import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("Clean_Dataset.csv")
X = data[['duration', 'days_left', 'class']]
y = data['price']

numeric_features = ['duration', 'days_left']
categorical_features = ['class']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(fill_value="missing", strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = {
    'model__n_estimators': [30, 40, 50]
    
}


pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', AdaBoostRegressor())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
mae = (mean_absolute_error(y_test, y_pred))
r2 = (r2_score(y_test, y_pred))
print("Ada Mean Abs Err: ", mae)
print("Ada R^2 Score: ", r2)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', AdaBoostRegressor())
])
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Ada w/ Grid Search Mean Abs Err: ", mae)
print("Ada w/ Grid Search R^2 Score: ", r2)


pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor())
])

param_grid = {
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Decision Tree w/ Grid Search Mean Abs Err: ", mae)
print("Decision Tree w/ Grid Search R^2 Score: ", r2)

base_estimator = DecisionTreeRegressor(max_depth=5)
adaboost_reg = AdaBoostRegressor(estimator=base_estimator, n_estimators=12, learning_rate=1.0)
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', adaboost_reg)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Decision Tree w/ Ada Boosting Mean Abs Err: ", mae)
print("Decision Tree w/ Ada Boosting R^2 Score: ", r2)