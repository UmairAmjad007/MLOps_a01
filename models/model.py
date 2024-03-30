import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['Year', 'Childrens Violent Discipline Rate']]
    y = df['Code']
    return LinearRegression().fit(X, y)
