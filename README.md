# Credit Card Fraud-Detection (AIML Project)

# Overview
This project leverages Artificial Intelligence and Machine Learning (AIML) technologies to detect fraudulent credit card transactions. By developing intelligent systems that can automatically identify potentially fraudulent transactions using transaction data, we aim to enhance the security of financial transactions and reduce financial losses due to fraud.

In this process, we have focused on analyzing and preprocessing data sets as well as using a supervised learning algorithm - Logistic Regression and Random Forest. This presents a comparative study of these two supervised learning algorithms.

# Purpose
1. Fraud Detection: Machine learning models to classify transactions as either legitimate or fraudulent based on transaction data.

# Technologies Used
1. Programming Languages: Python
2. Libraries: Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib
3. Development Environments: Google Colab, Jupyter Notebook
4. Algorithms: Logistic Regression, Random Forest

# Methodology
1. Data Collection: Gather transaction data including features such as transaction amount, time, and other relevant variables.
2. Data Preprocessing: Clean and preprocess the data to scale features, to balance the dataset.
3. Gather insights from data.
4. Model Training: Train machine learning models on the preprocessed data.
5. Evaluate the Model Accuracy: Assess the performance of the model by comparing the accuracy of training and testing data.

# Dataset Information
1. Time: Number of seconds elapsed between this transaction and the first transaction in the dataset.
2. V1, V2, ..., V28: Result of a Principal Component Analysis (PCA) transformation. Due to confidentiality issues, the original features are not provided.
3. Amount: Transaction amount.
4. Class: Class label, where 1 indicates a fraudulent transaction and 0 indicates a legitimate transaction.

# References
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Results
The results have revealed that both the models Logistic Regression and Random Forest achieved a high accuracy. However, there were several differences in their performance: 
1. Logistic Regression has achieved an accuracy of 92.39%, as the linear nature of the algorithm make it difficult to analyze complex patterns in the data resulting in comparatively lower accuracy.
2. In this experimentation, Random Forest has outperformed Logistic Regression with an accuracy of 99.95%. It allows all the decision tree to train on different subsets of the dataset resulting in higher accuracy. The non-linear nature of the algorithm makes it more flexible to tackle complex data.

# Conclusion
Through the experiments, we have shown that Random Forest offers higher accuracy concluding that Random Forest is better choice to handle nonlinear and complex data. 
