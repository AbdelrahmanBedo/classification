# README

## Introduction
This repository contains a Python script for performing machine learning tasks on a lung dataset using various classifiers and clustering algorithms. It covers data preprocessing, feature selection, model evaluation, and clustering analysis.

## Dependencies
Make sure you have the following dependencies installed:
- scipy
- pandas
- scikit-learn
- seaborn
- matplotlib



2. Navigate to the directory containing the script.

3. Run the script using Python:
```
python lung_ml_analysis.py
```

## Description
The script performs the following tasks:

1. Import necessary libraries:
```python
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
```

2. Load the ARFF file and convert to Pandas DataFrame:
```python
data, meta = arff.loadarff('C:\\Users\\Blu-Ray\\Desktop\\Lung.arff')
df = pd.DataFrame(data)
```

3. Explore the dataset:
```python
df.head()
df['type'].unique()
df.isnull().sum()
```

4. Data preprocessing:
```python
# Extract features (X) and target variable (y)
X = df.drop('type', axis=1)
y = df['type']

# Decode byte-encoded strings in the 'type' column
y_decoded = y.str.decode('utf-8')

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_decoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

5. Apply classifiers (K-Nearest Neighbors, Support Vector Machine, Neural Network) and evaluate the models:
```python
# Apply K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

# Apply Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# Apply Neural Network (NN)
nn_model = MLPClassifier()
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)

# Evaluate models
evaluate_model(knn_model, X_test_scaled, y_test, 'K-Nearest Neighbors (KNN)')
evaluate_model(svm_model, X_test_scaled, y_test, 'Support Vector Machine (SVM)')
evaluate_model(nn_model, X_test_scaled, y_test, 'Neural Network (NN)')
```

6. Perform feature selection and dimensionality reduction:
```python
# Apply Variance Threshold
selector_variance = VarianceThreshold(threshold=0.1)
X_high_variance = selector_variance.fit_transform(X)

# Apply Correlation Matrix
correlation_threshold = 0.8
...

# Apply Recursive Feature Elimination (RFE)
rfe_selector = RFE(RandomForestClassifier(), n_features_to_select=5)
X_rfe = rfe_selector.fit_transform(X_filtered, y_encoded)

# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_filtered, y_encoded)

# Apply Principal Component Analysis (PCA)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_filtered)

# Apply t-Distributed Stochastic Neighbor Embedding (t-SNE)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_filtered)
```

