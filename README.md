# Telco Customer Churn Prediction using SVM

This project is a complete machine learning pipeline built to predict **customer churn** using the popular **Telco Customer Churn Dataset**. It was developed as part of our final project for the Python Programming course (Fall 2024-25). The project includes essential steps such as data cleaning, visualization, feature scaling, model training, hyperparameter tuning, and evaluation using **Support Vector Machine (SVM)** classifier.

## 🔍 Project Objective

To accurately predict whether a telecom customer will churn (leave the company) based on their usage behavior and service attributes, helping businesses improve retention strategies.

---

## 📁 Dataset

* **Source**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Description**: This dataset contains information about a telecom company's customers, including services subscribed, account information, and whether the customer has churned. It is commonly used for predictive modeling and customer retention analysis.
* **Attributes**: Customer demographics, account info, service usage, and churn labels
* **Target Variable**: `Churn` (Binary Classification - Yes/No)

---

## 🧠 Machine Learning Pipeline

### ✅ Step 1: Data Loading

* Read the dataset from Google Drive using **Pandas**.

### 🧼 Step 2: Data Cleaning

* Converted incorrect data types.
* Handled missing values using **median imputation**.
* Removed duplicate entries to maintain data integrity.

### 📊 Step 3: Exploratory Data Analysis (EDA)

* Visualized **categorical feature distributions** using bar charts.
* Displayed **numerical feature distributions** using histograms.
* Plotted all graphs in a single figure using **Matplotlib’s subplot**.

### ⚙️ Step 4: Feature Encoding & Scaling

* Applied **Label Encoding** to categorical variables.
* Used **StandardScaler** to scale numerical features for better SVM performance.

### 🧪 Step 5: Train-Test Split

* Splitted data using `train_test_split()` with `random_state=3241` (80% train / 20% test).

### 🧮 Step 6: Model Training (SVM)

* Trained a baseline SVM classifier with a linear kernel.
* Performed **GridSearchCV** for hyperparameter tuning (`C`, `kernel`, `gamma`).
* Selected the best model based on **5-fold cross-validation accuracy**.

### 📈 Step 7: Model Evaluation

* Computed **Confusion Matrix** and **Classification Report**.
* Compared training vs testing accuracy to evaluate model performance.

---

## 📊 Results

* **Best Parameters**: Chosen via GridSearchCV
* **Train Accuracy**: \~**0.84**
* **Test Accuracy**: \~**0.79**
* **Confusion Matrix**: Provided in output for visual evaluation
* **Classification Metrics**: Precision, Recall, F1-score

---

## 🛠 Technologies Used

* Python 3.x
* Google Colab
* Pandas, NumPy
* Matplotlib
* Scikit-learn

---

## 👨‍🎓 Contributors

* **A. M. Rafinul Huq** — ID: 21-45668-3
* **Yeasir Ahnaf Asif** — ID: 20-42815-1

---


## ⌛ Project Timeline

* **Semester:** Fall 2024-2025
* **Date:** October 2024 - January 2025 
* **🔒 Note:** We intentionally uploaded the project to GitHub after the course presentation to avoid potential misuse or copying before our official evaluation.

---

## 📌 How to Run

1. Upload the `telco_churn.csv` dataset to Google Drive.
2. Open this project notebook/script in Google Colab.
3. Ensure all libraries are installed.
4. Run each cell step-by-step to reproduce results.

---

## 💼 Portfolio Use

This project is a great demonstration of:

* Real-world dataset handling
* End-to-end supervised learning
* Practical data preprocessing and model optimization
* SVM application in business decision-making

> Feel free to fork and build upon it for academic, career, or personal development purposes!

---

