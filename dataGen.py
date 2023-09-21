import os
from sdv.datasets.local import load_csvs
from sdv.lite import SingleTablePreset
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# get the current directory using os.path
folderName = os.path.abspath(os.path.dirname(__file__))

# load the csv file with the sample diabetes dataset
datasets = load_csvs(folder_name=folderName)
real_data = datasets["diabetes"]

# generate metadata for the sample dataset
metadata = SingleTableMetadata()
metadata.detect_from_csv(filepath=folderName + "/diabetes.csv")

# print(metadata.to_dict())

real_data.head()
metadata.visualize()

# Initializing a SingleTablePreset object with the metadata and fitting the synthesizer and sampling with the real_data input.
synthesizer = SingleTablePreset(metadata, name="FAST_ML")
synthesizer.fit(data=real_data)

# Generating 500 rows of synthetic data using the synthesizer and saving it as a csv
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv("synthetic_diabetes.csv", index=False)
synthesizer.save("diabetes.pkl")
print("\nSynthetic data generated.\n")
print(synthetic_data.head())

# Preprocessing of the Synthetic Data

# Load the synthetic dataset
synthetic_data = pd.read_csv("synthetic_diabetes.csv")

# Handle missing values (if any) by replacing them with the mean
synthetic_data.fillna(synthetic_data.mean(), inplace=True)

# Normalizing the data
# for column in synthetic_data:
#     synthetic_data[column] = (synthetic_data[column] - synthetic_data[column].min()) / (
#         synthetic_data[column].max() - synthetic_data[column].min()
#     )

print("\nSynthetic data Preprocessed.\n")

# Exploratory Data Analysis of the Synthetic Data
print("\nEDA for Synthetic data.\n")

# Display basic statistics
print(synthetic_data.describe())
print("\n")

# Check data types and missing values
print(synthetic_data.info())
print("\n")

# Calculate and visualize correlations between numeric columns using cluster maps
plt.figure(figsize=(10, 6))
sns.heatmap(synthetic_data.corr(), cmap="RdBu", center=0, cbar=True, annot=True)
sns.clustermap(synthetic_data.corr(), cmap="RdBu", center=0, cbar=True, annot=True)
plt.show()

plt.figure(figsize=(10, 10))
sns.boxplot(x="Outcome", y="Glucose", data=synthetic_data)
plt.xlabel("Outcome")
plt.ylabel("Glucose")
plt.show()

# Evaluating the quality of the synthetic data
quality_report = evaluate_quality(real_data, synthetic_data, metadata)
quality_report.get_visualization("Column Shapes")

# Save the Synthetic Data after preprocessing and evaluation
synthetic_data.to_csv("synthetic_diabetes.csv", index=False)
synthesizer.save("diabetes.pkl")

# Loading a saved model
# synthesizer = SingleTablePreset.load(folderName + "/diabetes.pkl")
