# Titanic-Survival-Prediction
 ![Uploading image.png…]()

Titanic-Survival-Prediction
CodTech IT Solutions Internship – Data Science Task Documentation : Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic Survived or not.
Intern Information
Name :- Prajakta Satish Pawars
ID :- COD6991
Overview
Predicting survival on the Titanic is a classic machine learning problem often used for educational purposes and as an introduction to predictive modeling techniques. The goal is to predict whether a passenger survived or not based on various features such as age, gender, ticket class, fare, and so on.
Dataset Description
The Titanic dataset is well-documented and includes the following columns:
•	PassengerId: A unique identifier for each passenger.
•	Survived: Target variable (0 = Not Survived, 1 = Survived).
•	Pclass: Passenger class (1st, 2nd, or 3rd class).
•	Name: Passenger's name.
•	Sex: Gender of the passenger.
•	Age: Age of the passenger.
•	SibSp: Number of siblings/spouses aboard the Titanic.
•	Parch: Number of parents/children aboard the Titanic.
•	Ticket: Ticket number.
•	Cabin: Cabin number.
•	Embarked: Port where the passenger boarded (C = Cherbourg, Q = Queenstown, S = Southampton).
Getting Started
To run this project locally, follow these steps:
1.	Clone the repository to your local machine:
- https://github.com/Praju-16/Titanic-Survival-Prediction.git
- pip install -r requirements.txt
2.	Download the Titanic dataset (CSV file) from Kaggle and place it in the project directory.
3.	Run the Jupyter Notebook or Python script to explore the dataset, preprocess the data, build the model, and evaluate its performance.
Data Preprocessing
Data preprocessing is a crucial step in preparing the dataset for analysis. The following preprocessing steps are performed:
•	Handling missing values, including imputing missing ages and embarked locations.
•	Encoding categorical variables, such as gender and embarked location, into numerical format.
•	Removing unnecessary columns like "PassengerId," "Name," "Ticket," "Cabin," and "Fare."
Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is an essential phase to gain insights into the dataset. The following visualizations and analyses are conducted:
•	Distribution of survival to understand the class imbalance.
•	Age distribution to observe the age demographics of passengers.
•	Survival by passenger class to identify class-based survival trends.
•	Survival by gender to explore gender-based survival trends.
•	A correlation heatmap to assess feature relationships.
Model Building
A logistic regression model is constructed to predict passenger survival. Logistic regression is chosen for its simplicity and interpretability, making it a suitable starting point for this binary classification problem. However, alternative machine learning algorithms could be explored for comparison.
Evaluation
The model's performance is assessed using various metrics, including:
•	Accuracy: The proportion of correct predictions.
Output 
 ![image](https://github.com/Praju-16/Titanic-Survival-Prediction/assets/141834374/3e30cac6-9dc7-4376-9c68-cad0f2dd5196)


