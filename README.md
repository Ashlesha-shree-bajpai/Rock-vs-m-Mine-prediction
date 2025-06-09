# Rock vs Mine Prediction using Logistic Regression

This repository contains a basic machine learning project that uses **Logistic Regression** to classify sonar signals as either **rock** or **mine** based on the Sonar dataset from the UCI Machine Learning Repository.

## 🔍 Problem Statement

The goal of this project is to predict whether an object detected by a submarine's sonar is a **rock** or a **mine**. This is a **binary classification** problem, and logistic regression is an ideal algorithm for solving such problems due to its simplicity and efficiency.

## 📁 Dataset

- **Source:** UCI Machine Learning Repository – [Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
- **Description:** 208 instances with 60 numeric attributes. Each attribute represents energy in a specific frequency band, and the target label is either **'R' (Rock)** or **'M' (Mine)**.

## 🧰 Libraries Used

- `pandas` – For data loading and manipulation  
- `numpy` – For numerical operations  
- `scikit-learn` (`sklearn`) – For model building, training, and evaluation

## 🧠 Machine Learning Model

- **Algorithm:** Logistic Regression  
- **Reason:** Suitable for binary classification problems, easy to implement, and interpretable.

## 🔧 Project Structure

```bash
├── sonar.csv                  # Dataset file
├── rock_vs_mine_prediction.py # Python script for training and prediction
├── README.md                  # Project documentation

  


 
