🏠 House Rent Prediction using Linear Regression
📌 Project Overview

This project develops a Linear Regression model to predict house rent based on key features such as:

Size (square feet)

Bedrooms (number of rooms)

Location (city/area)

The model uses supervised machine learning with preprocessing, training, and evaluation steps to provide accurate predictions.

⚙️ Technologies Used

Python 3

Pandas → Data loading & preprocessing

NumPy → Numerical operations

Scikit-Learn → Model training & evaluation

Matplotlib & Seaborn → Data visualization

📊 Workflow

Load Dataset

Data is stored in data.csv containing columns: Size, Bedroom, Location, and Rent.

Preprocessing

Convert categorical Location into numerical format using One-Hot Encoding.

Split dataset into train (80%) and test (20%) sets.

Model Training

Train a Linear Regression model using scikit-learn.

Evaluation

Measure model performance using:

Mean Squared Error (MSE)

R² Score

Visualize predictions with an Actual vs Predicted Rent scatter plot.

📈 Example Output

Model Performance:

Mean Squared Error (MSE): 2345678.91  
R² Score: 0.87  


Visualization:
A scatter plot showing how closely predicted rents match actual rents.

▶️ How to Run

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn


Update the file path inside the code:

file_path = r"C:/Users/GEO/Downloads/August_workshop_python/data.csv"


Run the script:

python codexintern2.py

🚀 Future Improvements

Add more features (furnishing status, amenities, year built, etc.)

Try advanced models like Random Forest or Gradient Boosting

Build a web app (Flask/Streamlit) for interactive predictions
