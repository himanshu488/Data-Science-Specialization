import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Multi.csv")

# Feature Selection
x = df[["area", "bedrooms", "age"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(x, y)

# Check Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict
print("Predicted Price for [5000, 3, 5]:", model.predict([[5000, 3, 5]]))

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["area"], df["bedrooms"], y, color="red", s=50)
ax.set_xlabel("Area")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")
plt.title("House Prices based on Area and Bedrooms")
plt.show()
