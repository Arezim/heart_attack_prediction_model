import pandas as pd
import numpy as np

# Load the Excel file
training_path = 'set2_500_patients.xlsx'
test_path = 'set3_500_patients.xlsx'

training_df = pd.read_excel(training_path)
test_df = pd.read_excel(test_path)

# Exclude the first column from the DataFrame
training_df = training_df.iloc[:, 1:]
test_df = test_df.iloc[:, 1:]

# Handle NaN values
training_df = training_df.dropna()
test_df = test_df.dropna()

# Select only numeric columns for the correlation analysis
training_numeric_columns = training_df.select_dtypes(include=['float64', 'int64', 'bool'])
test_numeric_columns = test_df.select_dtypes(include=['float64', 'int64', 'bool'])

# Correlation analysis for heart_attack and accident (from training data)
correlation_matrix_heart_attack = training_numeric_columns.corrwith(training_numeric_columns['heart_attack'])
correlation_matrix_accident = training_numeric_columns.corrwith(training_numeric_columns['accident'])

heart_attack_corr = correlation_matrix_heart_attack.sort_values(ascending=False)
accident_corr = correlation_matrix_accident.sort_values(ascending=False)

# Vectorized probability indicator calculation on training data
heart_attack_prob_indicator_train = np.dot(training_numeric_columns.values, heart_attack_corr.values)
accident_prob_indicator_train = np.dot(training_numeric_columns.values, accident_corr.values)

# Assign the probability indicators to the training dataframe
training_df['heart_attack_prob_indicator'] = heart_attack_prob_indicator_train
training_df['accident_prob_indicator'] = accident_prob_indicator_train

# Get the range of the probability indicators on the training data
heart_attack_min = heart_attack_prob_indicator_train.min()
heart_attack_max = heart_attack_prob_indicator_train.max()

accident_min = accident_prob_indicator_train.min()
accident_max = accident_prob_indicator_train.max()

# Function to calculate accuracy for a given threshold
def calculate_accuracy_for_threshold(df, threshold, target_col, prob_col):
    df['prediction'] = df[prob_col] > threshold
    accuracy = (df['prediction'] == df[target_col]).mean()
    return accuracy

# Grid search for optimal heart attack threshold on training data
best_heart_attack_threshold = 0
best_heart_attack_accuracy = 0

# Search the threshold across the actual range of the probability indicators
for heart_attack_thresh in np.arange(heart_attack_min, heart_attack_max, (heart_attack_max - heart_attack_min) / 100):
    heart_attack_accuracy = calculate_accuracy_for_threshold(training_df, heart_attack_thresh, 'heart_attack',
                                                             'heart_attack_prob_indicator')

    if heart_attack_accuracy > best_heart_attack_accuracy:
        best_heart_attack_accuracy = heart_attack_accuracy
        best_heart_attack_threshold = heart_attack_thresh

# Grid search for optimal accident threshold on training data
best_accident_threshold = 0
best_accident_accuracy = 0

# Search the threshold across the actual range of the probability indicators
for accident_thresh in np.arange(accident_min, accident_max, (accident_max - accident_min) / 100):
    accident_accuracy = calculate_accuracy_for_threshold(training_df, accident_thresh, 'accident',
                                                         'accident_prob_indicator')

    if accident_accuracy > best_accident_accuracy:
        best_accident_accuracy = accident_accuracy
        best_accident_threshold = accident_thresh

# Apply the pre-computed correlations to the test data
# Vectorized probability indicator calculation on test data using training coefficients
heart_attack_prob_indicator_test = np.dot(test_numeric_columns.values, heart_attack_corr.values)
accident_prob_indicator_test = np.dot(test_numeric_columns.values, accident_corr.values)

# Assign the probability indicators to the test dataframe
test_df['heart_attack_prob_indicator'] = heart_attack_prob_indicator_test
test_df['accident_prob_indicator'] = accident_prob_indicator_test

# Calculate accuracy on test data using the thresholds found on training data
heart_attack_test_accuracy = calculate_accuracy_for_threshold(test_df, best_heart_attack_threshold, 'heart_attack',
                                                              'heart_attack_prob_indicator')

accident_test_accuracy = calculate_accuracy_for_threshold(test_df, best_accident_threshold, 'accident',
                                                          'accident_prob_indicator')

# Display the best thresholds, corresponding accuracies on training data, and accuracies on test data
print(f"Best Heart Attack Threshold (Training Data): {best_heart_attack_threshold}")
print(f"Heart Attack Prediction Accuracy (Training Data): {best_heart_attack_accuracy:.2%}")
print(f"Heart Attack Prediction Accuracy (Test Data): {heart_attack_test_accuracy:.2%}")

print(f"Best Accident Threshold (Training Data): {best_accident_threshold}")
print(f"Accident Prediction Accuracy (Training Data): {best_accident_accuracy:.2%}")
print(f"Accident Prediction Accuracy (Test Data): {accident_test_accuracy:.2%}")
