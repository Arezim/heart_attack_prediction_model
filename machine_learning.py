import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# Load the Excel file
training_path = 'set2_500_patients.xlsx'
test_path = 'set3_500_patients.xlsx'

training_df = pd.read_excel(training_path)
test_df = pd.read_excel(test_path)

# Exclude the first column from the DataFrame
training_df = training_df.iloc[:, 1:-1]
test_df = test_df.iloc[:, 1:-1]

# Handle NaN values
training_df = training_df.dropna()
test_df = test_df.dropna()

# Select only numeric columns for the logistic regression
training_numeric_columns = training_df.select_dtypes(include=['float64', 'int64', 'bool'])
test_numeric_columns = test_df.select_dtypes(include=['float64', 'int64', 'bool'])

# Separate features and target for heart_attack prediction
X_train_heart_attack = training_numeric_columns.drop(columns=['heart_attack', 'accident'])
y_train_heart_attack = training_numeric_columns['heart_attack']

X_test_heart_attack = test_numeric_columns.drop(columns=['heart_attack', 'accident'])
y_test_heart_attack = test_numeric_columns['heart_attack']

# Train logistic regression model for heart attack prediction using only training data
logreg_heart_attack = LogisticRegression(max_iter=10000)
logreg_heart_attack.fit(X_train_heart_attack, y_train_heart_attack)

# Make predictions on both training and test sets (without using test data for training)
y_train_pred_heart_attack = logreg_heart_attack.predict(X_train_heart_attack)
y_test_pred_heart_attack = logreg_heart_attack.predict(X_test_heart_attack)

# Calculate accuracy on training and test sets for heart attack
heart_attack_train_accuracy = accuracy_score(y_train_heart_attack, y_train_pred_heart_attack)
heart_attack_test_accuracy = accuracy_score(y_test_heart_attack, y_test_pred_heart_attack)

# Calculate precision (True Positive Rate in relation to all positive results) for heart attack
heart_attack_train_precision = precision_score(y_train_heart_attack, y_train_pred_heart_attack)
heart_attack_test_precision = precision_score(y_test_heart_attack, y_test_pred_heart_attack)

# Repeat the same process for accident prediction
X_train_accident = training_numeric_columns.drop(columns=['heart_attack', 'accident'])
y_train_accident = training_numeric_columns['accident']

X_test_accident = test_numeric_columns.drop(columns=['heart_attack', 'accident'])
y_test_accident = test_numeric_columns['accident']

# Train logistic regression model for accident prediction using only training data
logreg_accident = LogisticRegression(max_iter=10000)
logreg_accident.fit(X_train_accident, y_train_accident)

# Make predictions on both training and test sets (without using test data for training)
y_train_pred_accident = logreg_accident.predict(X_train_accident)
y_test_pred_accident = logreg_accident.predict(X_test_accident)

# Calculate accuracy on training and test sets for accident
accident_train_accuracy = accuracy_score(y_train_accident, y_train_pred_accident)
accident_test_accuracy = accuracy_score(y_test_accident, y_test_pred_accident)

# Calculate precision (True Positive Rate in relation to all positive results) for accident
accident_train_precision = precision_score(y_train_accident, y_train_pred_accident)
accident_test_precision = precision_score(y_test_accident, y_test_pred_accident)

# Display results
print(f"Heart Attack Prediction Accuracy (Training Data): {heart_attack_train_accuracy:.2%}")
print(f"Heart Attack Prediction Accuracy (Test Data): {heart_attack_test_accuracy:.2%}")
print(f"Heart Attack Prediction Precision (Training Data): {heart_attack_train_precision:.2%}")
print(f"Heart Attack Prediction Precision (Test Data): {heart_attack_test_precision:.2%}")

print(f"Accident Prediction Accuracy (Training Data): {accident_train_accuracy:.2%}")
print(f"Accident Prediction Accuracy (Test Data): {accident_test_accuracy:.2%}")
print(f"Accident Prediction Precision (Training Data): {accident_train_precision:.2%}")
print(f"Accident Prediction Precision (Test Data): {accident_test_precision:.2%}")
