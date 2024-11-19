
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a simple dataset
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance_rate': [50, 60, 70, 65, 80, 85, 75, 90, 95, 100],
    'score': [40, 50, 55, 60, 65, 70, 72, 78, 85, 90]
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['study_hours', 'attendance_rate']]
y = df['score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Predictions: {y_pred}")

# Predicting a new student's score
new_student = np.array([[7, 80]])  # Example input: 7 hours of study, 80% attendance
predicted_score = model.predict(new_student)
print(f"Predicted Score for the new student: {predicted_score[0]:.2f}")
