# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# Step 2: Load Dataset
df = pd.read_csv("Multi.csv")  # Make sure this CSV has columns: area, bedrooms, age, price

# Step 3: Separate Features and Target
x = df[["area", "bedrooms", "age"]]  # Independent Variables (Multivariate)
y = df["price"]                      # Dependent Variable

# Step 4: 2D Data Visualizations (Area vs Price)
plt.figure(1)
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df["area"], y, color="red", edgecolors="blue")
plt.grid(True)
plt.show()
# Step 4.1: Bedrooms vs Price
plt.figure(2)
plt.title("Bedrooms vs Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.scatter(df["bedrooms"], y, color="green", edgecolors="black")
plt.grid(True)
plt.show()

# Step 4.2: Age vs Price
plt.figure(3)
plt.title("Age vs Price")
plt.xlabel("Age of House")
plt.ylabel("Price")
plt.scatter(df["age"], y, color="orange", edgecolors="black")
plt.grid(True)
plt.show()
# Step 5: Model Building
model = LinearRegression()
model.fit(x, y)

# Step 6: Coefficients and Intercept
print("Coefficients:", model.coef_)   # Coefficients for [area, bedrooms, age]
print("Intercept:", model.intercept_)

# Step 7: Prediction on Custom Input
area = int(input("Enter Area (sqft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
age = int(input("Enter Age of House: "))
predicted_price = model.predict([[area, bedrooms, age]])
print("Predicted Price is: ₹", predicted_price[0])
