import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to convert 'Lac' and 'Cr' to numeric rupees
def convert_price(value):
    try:
        value = str(value).replace(',', '').strip().lower()
        if 'lac' in value:
            return float(value.replace('lac', '').strip()) * 1e5
        elif 'cr' in value:
            return float(value.replace('cr', '').strip()) * 1e7
        else:
            return float(value)
    except:
        return None

# Load dataset
data = pd.read_csv("house_prices (1).csv")

# Convert and clean columns
data['Amount(in rupees)'] = data['Amount(in rupees)'].apply(convert_price)
data['Price (in rupees)'] = data['Price (in rupees)'].apply(convert_price)

# Drop rows with missing values
data = data[['Amount(in rupees)', 'Price (in rupees)']].dropna()

# Print sample data
print("\n📋 Sample Dataset:")
print(data.head(10))

# Feature & Target
X = data[['Amount(in rupees)']]
y = data['Price (in rupees)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)

print("\n📊 Model Evaluation:")
print(f"➡ Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
print(f"➡ R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"➡ Intercept: {model.intercept_:.2f}")
print(f"➡ Coefficient: {model.coef_[0]:.10f}")

# Visualization
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted', linewidth=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Amount (in rupees)')
plt.ylabel('Price (in rupees)')
plt.legend()
plt.grid(True)
plt.show()

# User input for prediction
# User input for prediction
try:
    user_input = float(input("\n💡 Enter 'Amount (in rupees)' to estimate price (e.g., 5000000 for ₹50 Lakhs): "))
    if user_input < 0:
        print("❌ Amount cannot be negative.")
    else:
        predicted_price = model.predict(pd.DataFrame([[user_input]], columns=['Amount(in rupees)']))[0]

        # Scaling logic (if output is too low)
        if predicted_price < 10000:  # low value means scaling is needed
            predicted_price *= 1e5  # adjust scaling factor as per dataset

        print(f"✅ Predicted Final Price: ₹{predicted_price:,.2f}")
except ValueError:
    print("❌ Invalid input. Please enter a numeric value like 5000000.")