import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['Year', 'Childrens Violent Discipline Rate']]
    y = df['Code']  # Assuming 'Code' is the target variable
    model = LinearRegression()
    model.fit(X, y)
    return model
