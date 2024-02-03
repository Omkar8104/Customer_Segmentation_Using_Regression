import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import Label, Entry, Button, messagebox, filedialog
from PIL import Image, ImageTk  # Import PIL

# Load the dataset (same as in your code)
dataset = pd.read_csv('Final_data.csv.csv')

# Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)
X = dataset[['Age', 'Salary', 'previous_purchase']].values  # Include 'previous_purchase' as an input feature
Y = dataset['previous_purchase'].values  # Assuming 'Purchased' is the target variable

# Splitting Dataset into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
