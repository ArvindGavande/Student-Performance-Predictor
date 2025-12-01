# Student-Performance-Predictor
The Student Performance Prediction System is an AI-powered application designed to analyze academic and behavioral factors of students and predict their overall academic performance.
The goal of the project is to help educators, institutions, and parents identify students who may require additional support, enabling early intervention and better learning outcomes.

In this project, a dataset containing student details—such as attendance, marks in core subjects, study hours, homework completion rate, gender, age, and parental education—was used to build machine learning models. Two predictive models were developed:

Classification Model
Predicts the student’s performance category (High, Medium, Low) using a Random Forest Classifier.

Regression Model
Estimates the student’s final score using a Random Forest Regressor.

Before training, the data was preprocessed through feature scaling, one-hot encoding of categorical variables, and proper train-test splitting. Both models achieved strong accuracy and reliability due to the non-linear learning capabilities of Random Forest algorithms.

To make the system more interactive and user-friendly, a Streamlit web application was developed. Users can manually input student information or upload a CSV file to generate predictions. The app displays:

Predicted performance category

Estimated final score

Class-wise probability distribution

Batch predictions for multiple students

This system can assist teachers and institutions in monitoring student progress, planning personalized interventions, and improving academic outcomes through data-driven insights.
