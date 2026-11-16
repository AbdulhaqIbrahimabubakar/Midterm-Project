# Student Performance Prediction (ML Zoomcamp Midterm Project)

## 1. Problem Description
Educational institutions aim to identify students at risk of low final exam performance early so they can offer targeted academic support.

This project uses the **UCI Student Performance Dataset (`student-mat.csv`)** containing 395 secondary school students and 33 attributes (demographics, family background, study habits, school support, and prior period grades `G1` and `G2`). The objective is to build a regression model to predict a student's final mathematics grade (**`G3`**, scored from 0 to 20), evaluated using **Root Mean Squared Error (RMSE)**.

## 2. Repository Structure
* `student-mat.csv`: Raw dataset (semicolon-delimited).
* `notebook.ipynb`: End-to-end workflow including data preparation, cleaning, EDA, feature importance analysis, model selection, and hyperparameter tuning.
* `train.py`: Script to train the final Gradient Boosting Regressor model and export the vectorizer and model to `model.bin`.
* `predict.py`: Flask web service script that loads `model.bin` and exposes a `/predict` POST endpoint.
* `test_predict.py`: Client script for sending a sample student JSON payload to the prediction API.
* `requirements.txt`: Python package dependencies.
* `Dockerfile`: Container configuration for running the prediction service.

## 3. Dataset
The dataset (`student-mat.csv`) is included directly in this repository. It originates from the [UCI Machine Learning Repository — Student Performance Dataset](https://uci.edu).

## 4. Exploratory Data Analysis & Feature Importance Highlights
* **Target Distribution (`G3`):** Final grades span 0 to 20 with a mean of ~10.4. There is a notable cluster of `0` scores alongside a roughly normal distribution around 11–12.
* **Top Numerical Predictors:** Second-period grade (`G2`) and first-period grade (`G1`) show the strongest positive correlation with `G3`, followed by mother's education (`medu`) and `studytime`. Past class `failures` has the strongest negative correlation.
* **Key Categorical Factors:** Students intending to pursue higher education (`higher = yes`) achieve noticeably higher average grades than the global mean.

## 5. Model Selection & Hyperparameter Tuning
Four regression algorithms were trained and compared on a 60/20/20 train/validation/test split using RMSE:
* **Linear Regression (Baseline):** Val RMSE = `2.3747`
* **Ridge Regression (`alpha=100.0`):** Val RMSE = `2.2011`
* **Random Forest Regressor (`max_depth=8, min_samples_leaf=5`):** Val RMSE = `2.1669`
* **Gradient Boosting Regressor (`learning_rate=0.05, max_depth=2`):** Val RMSE = `2.0889` *(Selected Final Model)*

## 6. How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Final Model
```bash
python train.py
```
This trains the tuned Gradient Boosting model and saves the pipeline to `model.bin`.

### 3. Run the Prediction Web Service
Start the Flask API server:
```bash
python predict.py
```
In a separate terminal window, test the `/predict` endpoint:
```bash
python test_predict.py
```

### 4. Run with Docker
Build and run the containerized prediction service:
```bash
docker build -t student-grade-predictor .
docker run -it --rm -p 9696:9696 student-grade-predictor
```
