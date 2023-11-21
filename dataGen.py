import os
from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
import warnings
warnings.filterwarnings( "ignore" )

# Getting the current directory using os.path and loading the csv file with the sample diabetes dataset
folderName = os.getcwd()
datasets = load_csvs(folder_name=folderName)
real_data = datasets["diabetes"]

# Generating metadata for the sample dataset
metadata = SingleTableMetadata()
metadata.detect_from_csv(filepath=folderName + "/diabetes.csv")

# Visualizing the metadata and print it
real_data.head()
metadata.visualize()
print("\n")
print(metadata.to_dict())

# Initializing a SingleTablePreset object with the metadata and fitting the synthesizer and sampling with the real_data input.
synthesizer = SingleTablePreset(metadata, name="FAST_ML")
synthesizer.fit(data=real_data)

# Generating 500 rows of synthetic data using the synthesizer and saving it as a csv and the synthesizer as a pkl
rows = 500
synthetic_data = synthesizer.sample(num_rows=rows)
synthetic_data.to_csv("synthetic_diabetes.csv", index=False)
synthesizer.save("diabetes.pkl")
print("\nSynthetic data generated.\n")
print(synthetic_data.head())

from sdv.evaluation.single_table import evaluate_quality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing of the Synthetic Data

# Handle missing values (if any) by replacing them with the mean
synthetic_data.fillna(synthetic_data.mean(), inplace=True)

print("\nSynthetic data Preprocessed.\n")

# Exploratory Data Analysis of the Synthetic Data
print("\nEDA for Synthetic data.\n")

# Display the Outcomes and its mean
print(synthetic_data['Outcome'].value_counts())
print("\n")
print(synthetic_data.groupby('Outcome').mean())
print("\n")

# Display basic statistics
print(synthetic_data.describe())
print("\n")

# Check data types and missing values
print(synthetic_data.info())
print("\n")

# Calculate and visualize correlations between numeric columns using cluster maps and box plots using seaborn

# Calculate correlations
correlation_matrix = synthetic_data.corr()

# Plot clustermap
plt.figure(figsize=(10, 6))
sns.clustermap(correlation_matrix, cmap="RdBu", center=0, cbar=True, annot=True)
plt.title("Correlation Clustermap")
plt.show()

# Box plot
plt.figure(figsize=(10, 10))
sns.boxplot(x="Outcome", y="Glucose", data=synthetic_data)
plt.xlabel("Outcome")
plt.ylabel("Glucose")
plt.title("Box Plot of Glucose by Outcome")
plt.show()

# Evaluating the quality of the synthetic data using sdv
quality_report = evaluate_quality(real_data, synthetic_data, metadata)
quality_report.get_visualization("Column Shapes")

# Save the Synthetic Data and the Synthesizing Model after preprocessing and evaluation
synthetic_data.to_csv("synthetic_diabetes.csv", index=False)
synthesizer.save("diabetes.pkl")

# Separating the features and target
target = synthetic_data["Outcome"]
features = synthetic_data.drop(columns="Outcome", axis = 1)

# Normalization and Standardization

# Normalizing the data
# features = (features - features.min()) / (features.max() - features.min())

# Standardizing the data
features = (features - features.mean()) / features.std()

# Splitting the data into training (80%) and test (20%)
total_samples = len(features)
train_samples = int(0.8 * total_samples)

# Shuffle the indices to randomize the data
indices = np.arange(total_samples)
np.random.shuffle(indices)

# Split the indices into training and test sets
train_indices = indices[:train_samples]
test_indices = indices[train_samples:]

# Create training and test data
x_tr = features.iloc[train_indices]
y_tr = target.iloc[train_indices]
x_te = features.iloc[test_indices]
y_te = target.iloc[test_indices]
