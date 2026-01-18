# Titanic Survival Prediction — Machine Learning Project

A **production-ready machine learning project** that predicts whether a passenger survived the Titanic disaster based on demographic and travel-related features.  
This project follows **industry-standard ML workflow**, from data exploration to deployment with Flask and Render.

---

##  Project Overview

The sinking of the Titanic is one of the most famous shipwrecks in history. In this project, we build a **binary classification model** to predict passenger survival using machine learning.

The goal is to demonstrate:
- Proper ML project structuring
- Clean data preprocessing and feature engineering
- Model training, evaluation, and saving
- Deployment readiness (Flask + Render)

---

## Objectives

- Explore and understand the Titanic dataset
- Clean and preprocess raw data
- Engineer meaningful features
- Train and compare multiple ML models
- Evaluate model performance using standard metrics
- Deploy the trained model as a web application

---

##  Machine Learning Task

- **Problem Type:** Binary Classification
- **Target Variable:** `Survived` (0 = No, 1 = Yes)
- **Algorithms Used:**
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier (Best Model)

---

##  Project Structure

```
Titanic-Survival-Prediction/
│
├── app/
│   ├── model/
│   │   ├── titanic_model.pkl
│   │   └── preprocessor.pkl
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── titanic.csv
│   └── processed/
│       ├── titanic_feature_engineered.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── requirements.txt
├── render.yaml
└── README.md
```

---

## Dataset

- **Dataset:** Titanic Passenger Dataset
- **Source:** Kaggle Titanic Competition
- **Size:** 891 rows × 12 columns

### Key Features
- `Pclass` – Passenger class
- `Sex` – Gender
- `Age` – Age of passenger
- `SibSp` – Siblings / spouses aboard
- `Parch` – Parents / children aboard
- `Fare` – Ticket fare
- `Embarked` – Port of embarkation

---

##  Data Preprocessing

Performed in `02_data_preprocessing.ipynb`:

- Missing value handling
  - Age → median
  - Embarked → mode
- Outlier detection and capping (IQR method)
- Dropping irrelevant columns
- Encoding categorical variables
- Scaling numerical features
- Train/Test split

---

##  Feature Engineering

Implemented in `03_feature_engineering.ipynb`:

- `FamilySize` = SibSp + Parch + 1
- `IsAlone` indicator
- Title extraction from passenger names
- Title grouping (Rare titles)
- Age group binning
- Fare band binning

These features significantly improved model performance.

---

##  Model Training

Conducted in `04_model_training.ipynb`:

| Model | Description |
|------|------------|
| Logistic Regression | Baseline model |
| Decision Tree | Non-linear relationships |
| Random Forest | Best-performing model |

- Pipelines ensure consistent preprocessing
- Models compared using **accuracy**
- Best model saved for deployment

---

##  Model Evaluation

Evaluated in `05_model_evaluation.ipynb` using:

- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Score
- ROC Curve Visualization

The final model shows strong generalization performance and is suitable for deployment.

---

##  Deployment

- **Backend:** Flask
- **Frontend:** HTML + CSS
- **Hosting:** Render

Users can input passenger details via a web form and receive a **real-time survival prediction**.

---

##  Installation & Usage

### 1️ Clone the Repository
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Flask App Locally
```bash
python app.py
```

Open browser at:
```
http://127.0.0.1:5000
```

---

##  Future Improvements

- Hyperparameter tuning
- Cross-validation
- Model explainability (SHAP)
- Dockerization
- CI/CD pipeline

---

##  Author

**Akinmusire Oluwankorinola**  
📍 Lagos, Nigeria  
oluwankorinolaa@gmail.com

---

## ⭐ Final Notes

This project demonstrates **end-to-end machine learning development** with a strong focus on clarity, structure, and deployability.  


