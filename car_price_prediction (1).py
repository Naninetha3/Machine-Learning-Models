import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
data = pd.DataFrame({
    'horsepower': [120, 150, 200, 130, 180],
    'mileage': [50000, 30000, 60000, 45000, 35000],
    'brand_goodwill': [0.7, 0.9, 0.6, 0.8, 0.85],
    'price': [15000, 20000, 18000, 16000, 22000]
})

# Features and target
X = data[['horsepower', 'mileage', 'brand_goodwill']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction
new_car = [[150, 40000, 0.8]]  # Example: horsepower, mileage, brand_goodwill
prediction = model.predict(new_car)
print(f"Predicted car price: ${prediction[0]:,.2f}")