import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample Dataset
data = {
    'Square_Feet': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Bathrooms': [1, 2, 2, 3, 3],
    'Price': [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Square_Feet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Train Model
model = LinearRegression()
model.fit(X, y)

# 🔹 Take User Input
print("Enter House Details:")
sqft = float(input("Square Feet: "))
bedrooms = int(input("Number of Bedrooms: "))
bathrooms = int(input("Number of Bathrooms: "))

# Predict Price
new_house = [[sqft, bedrooms, bathrooms]]
predicted_price = model.predict(new_house)

print("\nPredicted House Price: ₹", round(predicted_price[0], 2))
