## Kullanılan Kütüphane Sürümleri
Streamlit: 1.16.0
Numpy: 1.26.3
Pandas: 2.1.4
Seaborn: 0.12.2
Scikit-learn: 1.2.2


# Streamlit-Breast-Cancer-Diagnostic-App

## Introduction 
This project aims to develop an interactive web application using Streamlit for predicting breast cancer diagnosis based on the Wisconsin Diagnostic Breast Cancer dataset. The application provides a user-friendly interface for users to input relevant data and obtain predictions regarding the likelihood of breast cancer diagnosis.  
## Dataset 
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which is publicly available and widely used for machine learning tasks. It contains features computed from digitized images of breast mass, which are used to predict whether a tumor is malignant or benign.
## Features
*Data Analysis*: Explore the dataset by viewing the first and last 10 rows, column names, and summary statistics.
*Data Preprocessing*: Clean the dataset by removing unnecessary columns, mapping categorical values, removing correlated features, and scaling numerical features.
*Model Implementation*: Choose from different machine learning models including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Gaussian Naive Bayes, and Bernoulli Naive Bayes. Optimize the models using GridSearchCV.
*Model Analysis*: Evaluate the models using accuracy, precision, recall, and F1-score metrics. Display average scores and confusion matrix.
## Requirements
Streamlit==1.16.0
Numpy==1.26.3
Pandas==2.1.4
Seaborn==0.12.2
Scikit-learn==1.2.2
## How to Use
Install the required dependencies: pip install -r requirements.txt
Run the Streamlit app: streamlit run app.py
Upload a CSV file containing breast cancer data.
Explore the data, choose a machine learning model, and view the results.
## Contributors
Tugay Talha İçen (https://github.com/Tugaytalha)
## License
This project is licensed under the MIT License - see the LICENSE file for details.