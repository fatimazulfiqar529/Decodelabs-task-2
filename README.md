This project is part of the DecodeLabs AI Industrial Training Program (Batch 2026). It demonstrates a complete supervised machine learning pipeline for Data Classification using the K-Nearest Neighbors (KNN) algorithm on the classic Iris Benchmark Dataset.
The goal was to move beyond heuristic rule-based logic and implement a model that learns patterns from data to make intelligent classification decisions.
Project Goals:
Load and explore a real-world dataset
Apply feature scaling to remove bias
Split data into training and testing sets
Train a KNN classification model
Evaluate the model using a Confusion Matrix and F1 Score
Metric                Value
Algorithm       K-Nearest Neighbors (K=5)
Accuracy              100.00%
F1 Score              100.00%
Correct Predictions    30/30
Incorrect Predictions   0/30
Language: Python 3.x
Libraries:
scikit-learn: KNN model, StandardScaler, metrics
matplotlib: Visualization dashboard
numpy: Numerical operations
pandas: Data handling and display
Install dependencies:
pip install scikit-learn matplotlib numpy pandas
Run the script:
python Data-classification.py
