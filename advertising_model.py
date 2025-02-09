import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_advertising_data.csv")  

# Define features (X) and target variable (y)
X = df[["TV", "Radio", "Newspaper"]]  # 
y = df["Sales"]  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print model performance
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Save the trained model
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel training completed! The trained model is saved as 'linear_regression_model.pkl'.")
