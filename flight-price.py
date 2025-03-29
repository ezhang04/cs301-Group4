import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

data = pd.read_csv("Clean_Dataset.csv")
X = data[['duration', 'days_left', 'class']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

numeric_features = ['duration', 'days_left']
categorical_features = ['class']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),#strategy='median'
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),#strategy='most_frequent'
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the pipeline with preprocessing and model
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Make predictions
y_test_pred = pipe.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_test_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("R2 Score: ", r2_score(y_test, y_test_pred))