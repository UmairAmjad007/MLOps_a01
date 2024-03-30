import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_linear_regression_model(dataset_file):
    dataset_file = "dataset.csv"
    df = pd.read_csv(dataset_file)

    X = df[['Year']]  
    y = df['Movie ID']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    return model

if __name__ == "__main__":
    dataset_file = "data/movie_data.csv"
    model = train_linear_regression_model(dataset_file)
