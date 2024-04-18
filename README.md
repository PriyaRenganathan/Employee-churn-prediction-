# Employee-churn-prediction-
```python
import numpy as np

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define predict function
def predict(X, theta):
    prob = sigmoid(np.dot(X, theta))
    return prob.round().astype(int)

# Define input data
input_data = {
    'satisfaction_level': [0.38, 0.8, 0.11, 0.72],
    'last_evaluation': [0.53, 0.86, 0.88, 0.87],
    'number_project': [2, 5, 7, 5],
    'average_montly_hours': [157, 262, 272, 223],
    'time_spend_company': [3, 6, 4, 5],
    'Work_accident': [0, 0, 0, 0],
    'promotion_last_5years': [0, 0, 0, 0],
    'salary': ['low', 'high', 'high', 'medium']
}

# Convert salary to numerical
input_data['salary'] = [1 if sal == 'high' else 0 for sal in input_data['salary']]

# Add bias term to input data
X_input = np.array([[
    1, 
    input_data['satisfaction_level'][i],
    input_data['last_evaluation'][i],
    input_data['number_project'][i],
    input_data['average_montly_hours'][i],
    input_data['time_spend_company'][i],
    input_data['Work_accident'][i],
    input_data['promotion_last_5years'][i],
    input_data['salary'][i]
] for i in range(len(input_data['satisfaction_level']))])

# Define theta (weights)
theta = np.array([-1.0856306 ,  0.85834006,  0.8460897 , -0.22413186,  0.00434413,
       -0.07457676, -1.44748571, -0.55000402, -0.38606255])

# Predict churn for input data
predictions = predict(X_input, theta)
print("Churn predictions for input data:", predictions)
```
