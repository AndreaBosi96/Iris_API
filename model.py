# Importing necessary libraries
import pandas as pd  # To manage data as data frames
import numpy as np  # To manipulate data as arrays
from sklearn.linear_model import LogisticRegression  # Classification model


class Model:
    def __init__(self):
        # Importing the dataset
        df = pd.read_csv("./iris.csv")
        # Dictionary containing the mapping
        self.variety_mappings = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        # Encoding the target variables to integers
        df = df.replace(["Setosa", "Versicolor", "Virginica"], [0, 1, 2])
        X = df.iloc[:, 0:-1]  # Extracting the features/independent variables
        y = df.iloc[:, -1]  # Extracting the target/dependent variable
        self.logreg = LogisticRegression(
            max_iter=1000
        )  # Initializing the Logistic Regression model
        self.logreg.fit(X, y)  # Fitting the model

    # Function for classification based on inputs
    def classify(self, a, b, c, d):
        arr = np.array([a, b, c, d])  # Convert to numpy array
        arr = arr.astype(np.float64)  # Change the data type to float
        query = arr.reshape(1, -1)  # Reshape the array
        prediction = self.variety_mappings[
            self.logreg.predict(query)[0]
        ]  # Retrieve from dictionary
        return prediction  # Return the prediction
