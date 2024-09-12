import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Make a Data array with 4 sets of data 
# 1st value means Age
# 2nd value means Time spent
# 3rd value means Yes[1] or No[0]
X = np.array([[25, 30, 0], [30, 40, 1], [20, 35, 0], [35, 45, 1]])
Y = np.array([0, 1, 0, 1])  # Target labels (Yes[1] or No[0])

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the logistic regression model
Model = LogisticRegression()

# Train the model
Model.fit(X_train, Y_train)

# Check the accuracy (Optional)
Accuracy = Model.score(X_test, Y_test)
print(f"Model Accuracy: {Accuracy}")
#User input
User_age = float(input("Enter Customer Age"))
User_Timespend = float(input("Enter a time that you spended on site"))
User_Added_cart = int(input("Enter 1 if added to cart,Else Enter 0"))
User_array = np.array([[User_age,User_Timespend,User_Added_cart]])
prediction = Model.predict(User_array)
if prediction == 1:
    print("I think You likely to purchase How i can help you sir")
else:
    print("I think You  Unlikely to Purchase thanks for Visiitng")
    