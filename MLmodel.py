import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess your dataset
dataset = pd.read_csv('Final_data.csv.csv')
# ... (data preprocessing code)

# Segregate Dataset into X and Y
X = dataset[['Age', 'Salary', 'previous_purchase']].values
Y = dataset['previous_purchase'].values

# Splitting Dataset into Train, Validation, and Test Sets
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

# Hyperparameter Tuning (Grid Search Example)
param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_search = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
y_val_pred = best_model.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_val_pred)

# You can repeat the above steps with different models and configurations if needed
# ...

# Finally, evaluate the selected best model on the test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Validation Accuracy: {validation_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
